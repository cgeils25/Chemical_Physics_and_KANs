import sys
import os
sys.path.append(os.getcwd()) # add cwd to path so that we can import from utils

import numpy as np
import torch
import polars as pl
import matplotlib.pyplot as plt
import rdkit
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles
from utils.data_utils import get_all_descriptors_from_smiles_list
from utils.evaluation_utils import regression_report
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler # robust to outliers / skewness
from kan import KAN
from sklearn.ensemble import RandomForestRegressor

# This notebook will consist of some experiments seeing how KANs perform with a scaffold split vs a regular test split. The basic idea of a scaffold split is to ensure that there are no two molecules the train and test sets respectively that have the same molecular scaffold. The reasoning for doing this is that molecules with the same scaffold tend to have similar molecular properties, and therefore it becomes possible for the model (especially large neural nets) to simply memorize a lookup table of molecular scaffolds and their corresponding properties. 

options = {
    'random_seed': 1738,
    'device': 'cpu',
    'test_size': 0.2,
    'n_training_steps': 100,
}

options

filepath = 'datasets/aqueous_solubility_delaney.csv'
df_delaney = pl.read_csv(filepath)

# get list of smiles strings
smiles_list = list(df_delaney['SMILES'])

# compute all molecular descriptors and store in pandas dataframe
descriptors_df = get_all_descriptors_from_smiles_list(smiles_list, as_polars=True)

descriptors_df.head()

mols = list(map(rdkit.Chem.MolFromSmiles, smiles_list))

len(mols)

scaffolds = pl.Series(name='Saffold_SMILES', values=[MurckoScaffoldSmilesFromSmiles(smi) for smi in smiles_list])
scaffolds = scaffolds.replace("", "No_Scaffold")
scaffolds.value_counts(sort=True)

top_scaffolds = scaffolds.filter(scaffolds != "No_Scaffold").value_counts(sort=True)[:16]
top_scaffolds_smiles = top_scaffolds['Saffold_SMILES']
top_scaffolds_mols = [rdkit.Chem.MolFromSmiles(smi) for smi in top_scaffolds_smiles]
top_scaffolds_counts = top_scaffolds['count']  

MolsToGridImage(top_scaffolds_mols, molsPerRow=4, subImgSize=(200, 200), 
                legends=[f'Saffold_SMILES: {smi}\n Count: {count}' for smi, count in zip(top_scaffolds_smiles, top_scaffolds_counts)])

# Split Data

X = descriptors_df.to_torch()
y = df_delaney['measured log(solubility:mol/L)'].to_torch().to(torch.float32).reshape(-1, 1)

X.shape, y.shape

group_splitter = GroupShuffleSplit(n_splits=1, test_size=options['test_size'], random_state=options['random_seed']) 
# note: the test size will not actually end up as what we want because it might not even be possible given the constraint of keeping the groups separate

train_idx, test_idx = next(group_splitter.split(descriptors_df, groups=scaffolds))

X_train_group_split, X_test_group_split = X[train_idx], X[test_idx]
y_train_group_split, y_test_group_split = y[train_idx], y[test_idx]

X_train_group_split.shape, X_test_group_split.shape, y_train_group_split.shape, y_test_group_split.shape

# this needs to be taken into account when comparing performance. May just need to repeat the experiment a bunch to get meaningful results.
# hopefully mean test size across trials will be close to intended test size
actual_test_size = X_test_group_split.shape[0] / (X_train_group_split.shape[0] + X_test_group_split.shape[0])

actual_test_size

# sanity check: did the group split actually keep the scaffolds separate?
train_scaffolds = scaffolds[train_idx].unique() 
test_scaffolds = scaffolds[test_idx].unique()

for train_scaffold in train_scaffolds:
    assert train_scaffold not in test_scaffolds, 'Group split failed: scaffolds are not separated properly'

X_train_regular_split, X_test_regular_split, y_train_regular_split, y_test_regular_split = \
train_test_split(X, y, test_size=actual_test_size, random_state=options['random_seed'])

X_train_regular_split.shape, X_test_regular_split.shape, y_train_regular_split.shape, y_test_regular_split.shape

# Scale Features

group_split_scaler = RobustScaler()
regular_split_scaler = RobustScaler()

X_train_group_split_scaled = torch.tensor(group_split_scaler.fit_transform(X_train_group_split), dtype=torch.float32, device=options['device'])
X_test_group_split_scaled = torch.tensor(group_split_scaler.transform(X_test_group_split), dtype=torch.float32, device=options['device'])

X_train_regular_split_scaled = torch.tensor(regular_split_scaler.fit_transform(X_train_regular_split), dtype=torch.float32, device=options['device'])
X_test_regular_split_scaled = torch.tensor(regular_split_scaler.transform(X_test_regular_split), dtype=torch.float32, device=options['device'])

# Set Up Datasets

def make_dataset(X_train, y_train, X_test, y_test):
    return {
        'train_input': X_train,
        'train_label': y_train,
        'test_input': X_test,
        'test_label': y_test
    }

group_split_dataset = make_dataset(X_train_group_split_scaled, y_train_group_split, X_test_group_split_scaled, y_test_group_split)
regular_split_dataset = make_dataset(X_train_regular_split_scaled, y_train_regular_split, X_test_regular_split_scaled, y_test_regular_split)

# Did I Mess Up the Data?

assert not any([torch.isnan(t).any() or torch.isinf(t).any() for t in group_split_dataset.values()])
assert not any([torch.isnan(t).any() or torch.isinf(t).any() for t in regular_split_dataset.values()])

# Fit KAN

# Try to be scientific about scaffold vs. non-scaffold split

num_features = descriptors_df.shape[1]  

print(f'Number of features: {num_features}')

model_group_split = KAN(width=[num_features, 1], device=options['device'])

results_group_split = model_group_split.fit(dataset=group_split_dataset, opt='LBFGS', steps=options['n_training_steps'], lr=0.01)

model_regular_split = KAN(width=[num_features, 1], device=options['device'])

breakpoint()
results_regular_split = model_regular_split.fit(dataset=regular_split_dataset, opt='LBFGS', steps=options['n_training_steps'])

list(model_regular_split.parameters())

y_test_group_split_pred =  model_group_split(X_test_group_split_scaled).detach()

regression_report(y_test_group_split, y_test_group_split_pred)

# Does Random Forest Work?

model = RandomForestRegressor(random_state=options['random_seed'])

model.fit(X_train_regular_split_scaled.cpu().numpy(), y_train_regular_split.cpu().numpy().ravel())

y_test_regular_split_pred = model.predict(X_test_regular_split_scaled.cpu().numpy())

regression_report(y_test_regular_split, torch.tensor(y_test_regular_split_pred, device=options['device']))

model = RandomForestRegressor(random_state=options['random_seed'])

model.fit(X_train_group_split_scaled.cpu().numpy(), y_train_group_split.cpu().numpy().ravel())

y_test_group_split_pred = model.predict(X_test_group_split_scaled.cpu().numpy())

regression_report(y_test_group_split, torch.tensor(y_test_group_split_pred, device=options['device']))

