import sys; sys.path.append('.')

from typing import List

import argparse

import numpy as np
import torch
import torch.nn.functional as F
import polars as pl
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import VarianceThreshold

from train import train_regression

from utils.data_utils import get_all_descriptors_from_smiles_list, get_scaffolds, DESCRIPTOR_NAMES
from utils.evaluation_utils import regression_report
from utils.misc_utils import print_args

DTYPE = torch.float32
# MAKE SURE TO CHANGE THE RANDOM SEED FOR EACH RUN

# this is dumb. Make prettier later
class BagOfKans():
    def __init__(self, models):
        self.models = models
    
    def __call__(self, X):
        return torch.stack([bootstrapped_model(X)[:, 0] for bootstrapped_model in self.models], dim=0).mean(dim=0)


def get_data(path: str):
    df = pl.read_csv(path)

    smiles_list = df['SMILES'].to_list()

    X = get_all_descriptors_from_smiles_list(smiles_list, as_polars = True)
    y = df['measured log(solubility:mol/L)']
    scaffolds = get_scaffolds(smiles_list)
    
    return X, y, scaffolds


def process_data(X: pl.DataFrame, y: pl.DataFrame, groups: List, test_size: float, random_seed: int, 
                 variance_threshold: float, dtype: torch.dtype, return_feature_names_out: bool = False):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)

    train_idx, test_idx = next(splitter.split(X=X, y=y, groups=groups))

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_test = X[test_idx]
    y_test = y[test_idx] 

    selector = VarianceThreshold(threshold=variance_threshold)

    X_train_selected = selector.fit_transform(X_train)
    X_test_selected = selector.transform(X_test)

    scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=dtype)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=dtype)

    y_train_tensor = torch.tensor(y_train, dtype=dtype)
    y_test_tensor = torch.tensor(y_test, dtype=dtype)

    if return_feature_names_out:
        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, selector.get_feature_names_out()

    else:
        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
    

def get_single_model():
    # might not need this. would just simplify the process of obtaining models with the same hyperparameters for both the single models and the bags of kans
    pass


def train_single_models():
    # return models
    pass

def train_bags_of_kans():
    # return models
    pass

def evaluate_models():
    # collect regression reports
    pass

def compare_models():
    # do a t-test
    pass

def save_models():
    pass

def save_results():
    pass

def main(args):
    print_args(args)

    X, y, scaffolds = get_data(args.data_path)

    X_train, y_train, X_test, y_test = process_data(X=X, y=y, groups=scaffolds, test_size=args.test_size, random_seed=args.random_seed, 
                                                    variance_threshold=args.variance_threshold, dtype=DTYPE)

    breakpoint()
    pass

def parse_args():
    parser = argparse.ArgumentParser(description='Run an experiment comparing a single KAN to a Multiple KANs trained with Bootstrap Aggregation (Bagging)')
    parser.add_argument('--data_path', type=str, help='Path to the dataset')
    parser.add_argument('--n_trials', type=int, help='Number of trials to run', default=100)
    parser.add_argument('--variance_threshold', type=float, help='Variance threshold for feature selection', default=0.1)
    parser.add_argument('--random_seed', type=int, help='Random seed for reproducibility', default=1738)
    parser.add_argument('--test_size', type=float, help='Test size for the train-test split', default=0.2)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
