from utils.evaluation_utils import regression_report
import numpy as np

def test_regression_report():
    y_true = np.array([3., -0.5, 2., 7.])
    y_pred = np.array([2.5, 0.0, 2., 8.])

    report = regression_report(y_true, y_pred)

    assert isinstance(report, dict), 'report is not a dictionary'
    assert 'R2' in report, 'R2 not in report'
    assert 'MSE' in report, 'MSE not in report'
    assert 'MAE' in report, 'MAE not in report'
    assert 'MAPE' in report, 'MAPE not in report'
    assert 'RMSE' in report, 'RMSE not in report'

    R2 = 1 - ((np.sum((y_pred - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)))
    assert report['R2'] == R2, f"R2: {report['R2']} != {R2}"

    MSE = np.mean((y_true - y_pred) ** 2)
    assert report['MSE'] == MSE, f"MSE: {report['MSE']} != {MSE}"

    MAE = np.mean(np.abs(y_true - y_pred))
    assert report['MAE'] == MAE, f"MAE: {report['MAE']} != {MAE}"

    MAPE = np.mean(np.abs((y_true - y_pred) / y_true))
    assert report['MAPE'] == MAPE, f"MAPE: {report['MAPE']} != {MAPE}"

    RMSE = np.sqrt(np.mean((y_true - y_pred) ** 2))
    assert report['RMSE'] == RMSE, f"RMSE: {report['RMSE']} != {RMSE}"
