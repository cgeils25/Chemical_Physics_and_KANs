from utils.data_utils import get_all_descriptors_from_smiles_list, get_scaffolds
import numpy as np
import pandas as pd
import polars as pl

def test_get_all_descriptors_from_smiles_list_empty():
    # empty smiles list
    smiles_list = []
    result = get_all_descriptors_from_smiles_list(smiles_list, as_pandas=False)

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 0

    result = get_all_descriptors_from_smiles_list(smiles_list, as_pandas=True)

    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 0

    result = get_all_descriptors_from_smiles_list(smiles_list, as_pandas=False, as_polars=True)
    
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == 0

def test_get_all_descriptors_from_smiles_list_single():
    smiles_list = ["CCO"]
    result = get_all_descriptors_from_smiles_list(smiles_list, as_pandas=False)

    # check type
    assert isinstance(result, np.ndarray)

    # check shape
    assert result.shape[0] == 1

    result = get_all_descriptors_from_smiles_list(smiles_list, as_pandas=True)

    # check type
    assert isinstance(result, pd.DataFrame)

    # check shape
    assert result.shape[0] == 1

    result = get_all_descriptors_from_smiles_list(smiles_list, as_pandas=False, as_polars=True)

    # check type
    assert isinstance(result, pl.DataFrame)

    # check shape
    assert result.shape[0] == 1

def test_get_all_descriptors_from_smiles_list_dataset():
    """
    Test the function with a dataset of SMILES strings and precalculated molecular descriptors.

    A note about the test data: it was calculated using the following code:

    df_descriptors = pd.DataFrame()
    for i, mol in enumerate(tqdm(mols)):
        mol_descriptors_dict = Descriptors.CalcMolDescriptors(mol)
        mol_descriptors_df = pd.DataFrame(mol_descriptors_dict, index=[i])
        df_descriptors = pd.concat([df_descriptors, mol_descriptors_df], axis = 0)

    Also, it was done using the smiles strings from John Delaney's aqueous solubility dataset

    This is equivalent to what happens in function get_all_descriptors_from_smiles_list

    So, the point of this test is really to help with changes in versions of RDKit, i.e., if the methods for calculating molecular 
    descriptors change, this test will fail.
    """
    # load SMILES strings
    smiles_list = pd.read_csv('tests/test_datasets/delaney_smiles_test.csv').SMILES.to_list()

    # load precalculated descriptors
    precalculated_descriptors_df = pd.read_csv('tests/test_datasets/delaney_molecular_descriptors_test.csv')
    precalculated_descriptors = precalculated_descriptors_df.to_numpy()

    result_np = get_all_descriptors_from_smiles_list(smiles_list, as_pandas=False)
    result_df = get_all_descriptors_from_smiles_list(smiles_list, as_pandas=True)
    result_pl = get_all_descriptors_from_smiles_list(smiles_list, as_pandas=False, as_polars=True)

    # check types
    assert isinstance(result_np, np.ndarray)
    assert isinstance(result_df, pd.DataFrame)
    assert isinstance(result_pl, pl.DataFrame)

    # check shapes against smiles list
    assert result_np.shape[0] == len(smiles_list), 'Number of rows in numpy array is not equal to number of SMILES strings'
    assert result_df.shape[0] == len(smiles_list), 'Number of rows in dataframe is not equal to number of SMILES strings'
    assert result_pl.shape[0] == len(smiles_list), 'Number of rows in polars dataframe is not equal to number of SMILES strings'

    # check shapes against precalculated descriptors
    assert result_np.shape == precalculated_descriptors.shape, 'Shape of numpy array is not equal to shape of precalculated descriptors'
    assert result_df.shape == precalculated_descriptors_df.shape, 'Shape of dataframe is not equal to shape of precalculated descriptors'
    assert result_pl.shape == precalculated_descriptors_df.shape, 'Shape of polars dataframe is not equal to shape of precalculated descriptors'

    # check if values are equal
    assert np.allclose(result_np, precalculated_descriptors), "Numpy arrays of molecular descriptors are not equal"
    assert np.allclose(result_df, precalculated_descriptors_df), "Dataframes of molecular descriptors are not equal"
    assert np.allclose(result_pl, precalculated_descriptors_df), "Polars dataframes of molecular descriptors are not equal"


def test_get_scaffolds():
    smis = ['CCO', 'CC1=CC=CC=C1', 'OCC1=CN=CN=C1']
    scaffolds_ground_truth = ['no_scaffold_0', 'c1ccccc1', 'c1cncnc1']

    scaffolds = get_scaffolds(smis)

    assert len(scaffolds) == len(smis), f'Number of scaffolds is not equal to number of SMILES strings. Got {len(scaffolds)}, expected {len(smis)}'
    assert scaffolds == scaffolds_ground_truth, f'Scaffolds are not equal to ground truth. Got {scaffolds}, expected {scaffolds_ground_truth}'
