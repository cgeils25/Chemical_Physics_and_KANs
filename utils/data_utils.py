import numpy as np
import pandas as pd
import polars as pl
from rdkit import Chem
from rdkit.Chem import Descriptors, GraphDescriptors
from tqdm import tqdm

import warnings

# suppresses a faulty deprecation warning from rdkit
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 
# warning raised by Descriptors.CalcMolDescriptors -- Maintainers claim it will be fixed in 2024.03.06

def get_all_descriptors_from_smiles_list(smiles_list: list[str], as_pandas: bool = False, as_polars: bool = False, show_tqdm: bool = True):
    """Calculates all molecular descriptors from a list of SMILES strings.

    Args:
        smiles_list (list[str]): List of SMILES strings.
        as_pandas (bool, optional): If True, returns a pandas dataframe. Defaults to False.
        as_polars (bool, optional): If True, returns a polars dataframe. Defaults to False.
        show_tqdm (bool, optional): If True, shows a progress bar. Defaults to True.

    Returns:
        np.ndarray or pd.DataFrame: Array or pandas dataframe of molecular descriptors 
    """
    # input validation
    if not isinstance(smiles_list, list):
        raise ValueError(f"smiles_list must be a list, instead got {type(smiles_list)}")
    
    if not all(isinstance(smiles, str) for smiles in smiles_list):
        raise ValueError("All elements in smiles_list must be strings")
    
    if not isinstance(as_pandas, bool):
        raise ValueError(f"as_pandas must be a boolean, instead got {type(as_pandas)}")

    if not isinstance(as_polars, bool):
        raise ValueError(f"as_polars must be a boolean, instead got {type(as_polars)}")
    
    if as_pandas and as_polars:
        as_polars = False
        warnings.warn("Both as_pandas and as_polars are True. Returning pandas dataframe")

    # if input smiles list is empty, return empty array or dataframe
    if len(smiles_list) == 0:
        if as_pandas:
            return pd.DataFrame()
        if as_polars:
            return pl.DataFrame()
        return np.array([])
    
    # get names of molecular descriptors
    descriptor_names = [x[0] for x in Descriptors._descList]
    IpC_idx = descriptor_names.index('Ipc') # so I can replace it with the avergae IpC. Otherwise it spits out numbers > 1e50

    # initialize empty array to store descriptors
    all_descriptors = np.empty((len(smiles_list), len(descriptor_names)))

    for i, smiles in enumerate(tqdm(smiles_list, desc="Calculating descriptors")) if show_tqdm else enumerate(smiles_list):
        # obtain mol object
        mol = Chem.MolFromSmiles(smiles)

        # calculate all descriptors for the molecule
        mol_descriptors = np.array(list(Descriptors.CalcMolDescriptors(mol).values()))

        # replace Ipc with average IpC. Otherwise it spits out numbers > 1e50. See https://www.rdkit.org/docs/source/rdkit.Chem.GraphDescriptors.html
        mol_descriptors[IpC_idx] = GraphDescriptors.Ipc(mol, avg=True)

        # store descriptors in array
        all_descriptors[i] = mol_descriptors
    
    # check if any values are NaN or infinite
    if np.isnan(all_descriptors).any():
        warnings.warn("some descriptors are NaN, check input SMILES strings")

    if np.isinf(all_descriptors).any():
        warnings.warn("some descriptors are infinite, check input SMILES strings")
    
    if as_pandas:
        # convert to pandas dataframe
        all_descriptors_df = pd.DataFrame(all_descriptors, columns=descriptor_names)
        return all_descriptors_df
    if as_polars:
        # convert to polars dataframe
        all_descriptors_df = pl.DataFrame(all_descriptors, schema=descriptor_names)
        return all_descriptors_df

    return all_descriptors

    