import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

# suppresses a faulty deprecation warning from rdkit
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 
# warning raised by Descriptors.CalcMolDescriptors -- Maintainers claim it will be fixed in 2024.03.06

def get_all_descriptors_from_smiles_list(smiles_list: list[str], as_dataframe: bool = False):
    """Calculates all molecular descriptors from a list of SMILES strings.

    Args:
        smiles_list (list[str]): List of SMILES strings.
        as_dataframe (bool, optional): If True, returns a pandas dataframe. Defaults to False.

    Returns:
        np.ndarray or pd.DataFrame: Array or pandas dataframe of molecular descriptors 
    """
    # input validation
    if not isinstance(smiles_list, list):
        raise ValueError(f"smiles_list must be a list, instead got {type(smiles_list)}")
    
    if not all(isinstance(smiles, str) for smiles in smiles_list):
        raise ValueError("All elements in smiles_list must be strings")
    
    if not isinstance(as_dataframe, bool):
        raise ValueError(f"as_dataframe must be a boolean, instead got {type(as_dataframe)}")
    
    # if input smiles list is empty, return empty array or dataframe
    if len(smiles_list) == 0:
        if as_dataframe:
            return pd.DataFrame()
        return np.array([])
    
    # get names of molecular descriptors
    descriptor_names = [x[0] for x in Descriptors._descList]

    # initialize empty array to store descriptors
    all_descriptors = np.empty((len(smiles_list), len(descriptor_names)))

    for i, smiles in enumerate(tqdm(smiles_list)):
        # obtain mol object
        mol = Chem.MolFromSmiles(smiles)

        # calculate all descriptors for the molecule
        mol_descriptors = np.array(list(Descriptors.CalcMolDescriptors(mol).values()))

        # store descriptors in array
        all_descriptors[i] = mol_descriptors
    
    if as_dataframe:
        # convert to pandas dataframe
        all_descriptors_df = pd.DataFrame(all_descriptors, columns=descriptor_names)
        return all_descriptors_df

    return all_descriptors

    