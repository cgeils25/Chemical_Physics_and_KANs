"""
A CLI-based pipeline to run an experiment comparing single KANs to a Bag of KANs model trained with Bootstrap Aggregation (Bagging) on the Delaney aqueous solubility dataset.

for more information on command line arguments, run `python full_experiments/run_bag_of_KANs_experiment.py --help`
"""

import sys; sys.path.append('.')

import os

from typing import List

import argparse

import numpy as np
import torch
import torch.nn.functional as F
import polars as pl
import matplotlib.pyplot as plt

from kan import KAN

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import VarianceThreshold

from train import train_regression

from utils.data_utils import get_all_descriptors_from_smiles_list, get_scaffolds, DESCRIPTOR_NAMES
from utils.evaluation_utils import regression_report
from utils.misc_utils import print_args

from neattime import neattime

DTYPE = torch.float32
# MAKE SURE TO CHANGE THE RANDOM SEED FOR EACH RUN

# this is dumb. Make prettier later
class BagOfKans():
    def __init__(self, models):
        self.models = models
    
    def __call__(self, X):
        return torch.stack([bootstrapped_model(X)[:, 0] for bootstrapped_model in self.models], dim=0).mean(dim=0)
    
    def predict_and_return_individual_predictions(self, X):
        individual_predictions = torch.stack([bootstrapped_model(X)[:, 0] for bootstrapped_model in self.models], dim=0)

        return individual_predictions.mean(dim=0), individual_predictions


def get_delaney_aqueous_solubility_data(path: str):
    """Loads the Delaney aqueous solubility dataset from a csv file and returns the molecular descriptors, target values, and scaffolds.

    Args:
        path (str): Path to the csv file.

    Returns:
        pl.DataFrame, pl.DataFrame, List[str]: Returns the molecular descriptors, target values, and scaffolds.
    """
    df = pl.read_csv(path)

    smiles_list = df['SMILES'].to_list()

    X = get_all_descriptors_from_smiles_list(smiles_list, as_polars = True)
    y = df['measured log(solubility:mol/L)']
    scaffolds = get_scaffolds(smiles_list)
    
    return X, y, scaffolds


def process_data(X: pl.DataFrame, y: pl.DataFrame, groups: List, test_size: float, random_seed: int, 
                 variance_threshold: float, dtype: torch.dtype, device: torch.device, return_feature_names_out: bool = False):
    """Processes the data by splitting it into training and testing sets using a group split, selecting features, scaling the data, 
    and converting it to pytorch tensors.

    Args:
        X (pl.DataFrame): input features
        y (pl.DataFrame): target values
        groups (List): group identity for each sample, used for group split
        test_size (float): proportion of data in the test set. Note: final test set size may be different due to the group split 
        (it likely isn't possible to split the groups along and exact percentage)
        random_seed (int): random seed for reproducibility
        variance_threshold (float): variance threshold for feature selection
        dtype (torch.dtype): data type for the pytorch tensors
        device (torch.device): device for the pytorch tensors
        return_feature_names_out (bool, optional): whether to return the names of features that were retained after feature selection. Defaults to False.

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) or
        (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray): Returns the processed training and (optionally) the names of the selected features.
    """
    
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

    X_train_tensor = torch.tensor(X_train_scaled, dtype=dtype, device=device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=dtype, device=device)

    y_train_tensor = torch.tensor(y_train, dtype=dtype, device=device)
    y_test_tensor = torch.tensor(y_test, dtype=dtype, device=device)

    if return_feature_names_out:
        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, selector.get_feature_names_out()

    else:
        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


def train_and_test_predictions_to_report_df(y_train: torch.tensor, y_train_pred: torch.tensor, 
                                            y_test: torch.tensor, y_test_pred: torch.tensor, 
                                            trial_num: int, model_type: str):
    train_report = regression_report(y_true=y_train, y_pred=y_train_pred)
    test_report = regression_report(y_true=y_test, y_pred=y_test_pred)

    report_df_train = pl.DataFrame(data=train_report)
    report_df_train = report_df_train.with_columns(pl.lit('train').alias('train_or_test'))

    report_df_test = pl.DataFrame(data=test_report)
    report_df_test = report_df_test.with_columns(pl.lit('test').alias('train_or_test'))

    report_df = pl.concat([report_df_train, report_df_test])

    report_df = report_df.with_columns(
        pl.lit(trial_num).alias('trial_num'),
        pl.lit(model_type).alias('model_type')
    )

    return report_df


def train_single_model(X_train: torch.tensor, X_test: torch.tensor, y_train: torch.tensor, y_test: torch.tensor, 
                       num_itrs: int, lr: float, random_seed: int, trial_num: int, device: torch.device, verbose: bool = True):
    """Train a single KAN and return a report of final loss metrics as a polars DataFrame.

    Args:
        X_train (torch.tensor): train features
        X_test (torch.tensor): test features
        y_train (torch.tensor): train target values
        y_test (torch.tensor): test target values
        num_itrs (int): number of training iterations
        lr (float): learning rate
        random_seed (int): random seed for reproducibility
        trial_num (int): trial number
        device (torch.device): device for the pytorch tensors
        verbose (bool, optional): whether to print loss metrics as model trains. Defaults to True.

    Returns:
        pl.DataFrame: a DataFrame with the final loss metrics
    """
    
    single_model = KAN(width=[X_train.shape[1], 1], seed=random_seed, auto_save=False)

    single_model.to(device)
    
    train_regression(model=single_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, num_itrs=num_itrs, lr=lr, verbose=verbose, seed=random_seed, n_iterations_per_print=50)
    
    y_train_pred = single_model(X_train)[:, 0].detach().cpu().numpy()
    y_test_pred = single_model(X_test)[:, 0].detach().cpu().numpy()

    report_df = train_and_test_predictions_to_report_df(y_train=y_train.detach().cpu().numpy(), y_train_pred=y_train_pred,
                                                        y_test=y_test.detach().cpu().numpy(), y_test_pred=y_test_pred, trial_num=trial_num, 
                                                        model_type='single_kan')

    return report_df


def train_bags_of_kans(X_train: torch.tensor, X_test: torch.tensor, y_train: torch.tensor, y_test: torch.tensor, 
                       num_itrs: int, lr: float, num_bootstrap_samples: int, random_seed: int, trial_num: int, 
                       device: torch.device, verbose: bool = True):
    """Train a single KAN and return a report of final loss metrics as a polars DataFrame.

    Args:
        X_train (torch.tensor): train features
        X_test (torch.tensor): test features
        y_train (torch.tensor): train target values
        y_test (torch.tensor): test target values
        num_itrs (int): number of training iterations
        lr (float): learning rate
        num_bootstrap_samples (int): number of bootstrap samples and models
        random_seed (int): random seed for reproducibility
        trial_num (int): trial number
        device (torch.device): device for the pytorch tensors
        verbose (bool, optional): whether to print loss metrics as model trains. Defaults to True.

    Returns:
        pl.DataFrame: a DataFrame with the final loss metrics
    """
    
    torch_random_generator = torch.Generator().manual_seed(random_seed)

    bootstrapped_models = []
    
    for i in range(num_bootstrap_samples):
        if verbose:
            print('.'*100, f'\nTraining bootstrapped model {i+1}/{num_bootstrap_samples}')

        bootstrap_sample_indices = torch.randint(low=0, high=X_train.shape[0], 
                                                size=(X_train.shape[0],), 
                                                generator=torch_random_generator)

        X_train_bootstrap = X_train[bootstrap_sample_indices]
        y_train_bootstrap = y_train[bootstrap_sample_indices]

        bootstrapped_model = KAN(width=[X_train.shape[1], 1], seed=random_seed + (num_bootstrap_samples * i) # so that none will have the same random seed across the entire experiment
                                 , auto_save=False)
        
        bootstrapped_model.to(device)

        # not collecting reports for individual bootstrapped models
        train_regression(model=bootstrapped_model, X_train=X_train_bootstrap, y_train=y_train_bootstrap, X_test=X_test, y_test=y_test, num_itrs=num_itrs, lr=lr, verbose=verbose, seed=random_seed, n_iterations_per_print=50)

        bootstrapped_models.append(bootstrapped_model)
    
    bag_of_kans = BagOfKans(models=bootstrapped_models)

    y_train_pred = bag_of_kans(X_train).detach().cpu().numpy()
    y_test_pred = bag_of_kans(X_test).detach().cpu().numpy()

    report_df = train_and_test_predictions_to_report_df(y_train=y_train.detach().cpu().numpy(), y_train_pred=y_train_pred,
                                                        y_test=y_test.detach().cpu().numpy(), y_test_pred=y_test_pred, trial_num=trial_num, 
                                                        model_type='bag_of_kans')

    return report_df


def get_device(use_gpu: bool):
    """Get a device for pytorch tensors

    Args:
        use_gpu (bool): whether to search for a GPU

    Returns:
        torch.device: the device
    """
    if use_gpu:
        if torch.cuda.is_available():
            print('Using cuda gpu')
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            print('Using MPS gpu')
            return torch.device('mps')
        else:
            print('No GPU found. Using CPU.')
            return torch.device('cpu')
    
    print('Using CPU')
    return torch.device('cpu')


def main(args):
    print_args(args)

    device = get_device(args.use_gpu)
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if args.save_predictions:
        raise NotImplementedError('Saving predictions is not yet implemented. Would be useful for looking at the individual predictions of bagged kan')

    X, y, scaffolds = get_delaney_aqueous_solubility_data(args.data_path)

    result_df = None

    for trial_num in range(args.n_trials):
        print('-'*100, f'\nTrial {trial_num+1}/{args.n_trials}')

        print('Processing data...')
        X_train, y_train, X_test, y_test = process_data(X=X, y=y, groups=scaffolds, test_size=args.test_size, random_seed=args.random_seed + trial_num, 
                                                        variance_threshold=args.variance_threshold, dtype=DTYPE, device=device)
        print('Data processed.')

        print('Training single model...')
        single_model_result_df_trial = train_single_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, num_itrs=args.num_itrs, lr=args.lr, 
                                                          random_seed=args.random_seed + trial_num, trial_num=trial_num, device=device)
        print('Single model trained.')

        print('Training Bag of KANs...')
        bag_of_kans_result_df_trial = train_bags_of_kans(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, num_itrs=args.num_itrs, lr=args.lr, 
                                                         num_bootstrap_samples=args.num_bootstraps, random_seed=args.random_seed + trial_num, trial_num=trial_num,
                                                         device=device)
        print('Bag of KANs trained.')

        if result_df is None:
            result_df = pl.concat([single_model_result_df_trial, bag_of_kans_result_df_trial])

        else:
            result_df = pl.concat([result_df, single_model_result_df_trial, bag_of_kans_result_df_trial])
    
    print('All trials completed. Saving results...')

    result_filename = f'final_loss_data.csv'

    result_filepath = os.path.join(args.result_dir, result_filename)

    result_df.write_csv(result_filepath)

    print(f'Results saved to {result_filepath}')
    

def parse_args():
    parser = argparse.ArgumentParser(description='Run an experiment comparing a single KAN to a Multiple KANs trained with Bootstrap Aggregation (Bagging)')
    parser.add_argument('--data_path', type=str, help='Path to the dataset')
    parser.add_argument('--n_trials', type=int, help='Number of trials to run', default=100)
    parser.add_argument('--variance_threshold', type=float, help='Variance threshold for feature selection', default=0.1)
    parser.add_argument('--random_seed', type=int, help='Random seed for reproducibility', default=1738)
    parser.add_argument('--test_size', type=float, help='Test size for the train-test split', default=0.2)
    parser.add_argument('--num_itrs', type=int, help='Number of training iterations for both models', default=500)
    parser.add_argument('--lr', type=float, help='Learning rate for both models', default=0.01)
    parser.add_argument('--num_bootstraps', type=int, help='Number of bootstrap samples and models for the Bag of KANs model', default=50)
    parser.add_argument('--result_dir', type=str, help='Directory in which to save model training results', default=f'single_kan_vs_bag_of_kans_results/results_{neattime()}')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--save_predictions', action='store_true', help='Whether to save predictions for each trial')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
