"""
Builds a new test dataset with rdkit molecular descriptors for the delaney aqueous solubility dataset
"""

import pandas as pd
from tqdm import tqdm
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles

def main():
    smiles_df = pd.read_csv("tests/test_datasets/delaney_smiles_test.csv")

    smiles_list = smiles_df["SMILES"].to_list()

    mols = [MolFromSmiles(smiles) for smiles in smiles_list]

    df_descriptors = pd.DataFrame()
    for i, mol in enumerate(tqdm(mols)):
        mol_descriptors_dict = Descriptors.CalcMolDescriptors(mol)
        mol_descriptors_df = pd.DataFrame(mol_descriptors_dict, index=[i])
        df_descriptors = pd.concat([df_descriptors, mol_descriptors_df], axis = 0)
    
    df_descriptors.to_csv("tests/test_datasets/delaney_molecular_descriptors_test.csv", index=False)


if __name__ == "__main__":
    main()