"""
Utility functions for model evaluation.
"""
from typing import Union
import torch
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

def regression_report(y_true: Union[list, np.ndarray, torch.tensor], y_pred: Union[list, np.ndarray, torch.tensor]) -> dict:
    """Report regression metrics.

    Args:
        y_true (Union[list, np.ndarray, torch.tensor]): Ground truth (correct) target values.
        y_pred (Union[list, np.ndarray, torch.tensor]): Estimated targets as returned by a classifier.

    Returns:
        dict: Dictionary containing regression metrics.
    """
    report = {}

    report['R2'] = r2_score(y_true, y_pred)
    report['MSE'] = mean_squared_error(y_true, y_pred)
    report['MAE'] = mean_absolute_error(y_true, y_pred)
    report['MAPE'] = mean_absolute_percentage_error(y_true, y_pred)
    report['RMSE'] = root_mean_squared_error(y_true, y_pred)

    return report
