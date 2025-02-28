import torch
import torch.nn.functional as F

from kan import KAN

from train import train_regression

def test_train_regression():
    """
    Try to fit a KAN to target values produced by a randomly initialized model.
    """

    torch.manual_seed(1738)
    dummy_model = KAN(width=[3, 2, 1])

    X = torch.rand(size=(100, 3))
    y = dummy_model(X)[:, 0].detach()
    
    del dummy_model

    # train this model to match the randomly initialized dummy_model
    model = KAN(width=[3, 4, 4, 1])

    results = train_regression(model, X_train=X, y_train=y, X_test=X, y_test=y, num_itrs=100, lr=0.01, verbose=False)

    for metric, metric_values in results.items():
        assert isinstance(metric_values, torch.Tensor), f'{metric} is not a tensor'
        assert metric_values.dtype == torch.float32, f'{metric} dtype is not float32: got {metric_values.dtype}'
        assert metric_values.shape == (100,), f'{metric} shape is not correct: got {results["train_mses"].shape} but expected (100,)'
        assert torch.all(torch.isfinite(metric_values)), f'{metric} has non-finite values'
        assert torch.all(torch.isreal(metric_values)), f'{metric} has non-real values'
        assert not torch.any(torch.isnan(metric_values)), f'{metric} has NaN values'
    
    # did model learn to approximate dummy_model?
    assert results['train_mses'][-1] < 0.01, f"final train mse is too high: {results['train_mses'][-1]}"
    assert results['test_mses'][-1] < 0.01, f"final test mse is too high: {results['test_mses'][-1]}"

    assert results['train_r2s'][-1] > 0.95, f"final train r2 is too low: {results['train_r2s'][-1]}"
    assert results['test_r2s'][-1] > 0.95, f"final test r2 is too low: {results['test_r2s'][-1]}"

    assert results['train_maes'][-1] < 0.01, f"final train mae is too high: {results['train_maes'][-1]}"
    assert results['test_maes'][-1] < 0.01, f"final test mae is too high: {results['test_maes'][-1]}"

    assert results['train_rmses'][-1] < 0.01, f"final train rmse is too high: {results['train_rmses'][-1]}"
    assert results['test_rmses'][-1] < 0.01, f"final test rmse is too high: {results['test_rmses'][-1]}"
