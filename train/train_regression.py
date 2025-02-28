import torch
import torch.nn.functional as F
import warnings 


def train_regression(model: torch.nn.Module, X_train: torch.tensor , y_train: torch.tensor, X_test: torch.tensor, y_test: torch.tensor, 
                     lr: float = 0.01, num_itrs: int = 500, verbose: bool = True, seed: int = 1738, n_iterations_per_print: int = 50) -> dict:
    """Train a Kolmogorov Arnold Network (KAN) to minimize mean squared error (MSE). 

    This implementation does not batch data whatsoever; it simply trains and tests on the entire dataset. Not optimal for large datasets.

    Args:
        model (torch.nn.Module): a KAN model (from pykan) implemented in PyTorch
        X_train (torch.tensor): train features
        y_train (torch.tensor): train labels
        X_test (torch.tensor): test features
        y_test (torch.tensor): test labels
        lr (float, optional): learning rate. Defaults to 0.01.
        num_itrs (int, optional): number of training iterations (epochs). Defaults to 500.
        verbose (bool, optional): whether to print losses during training. Defaults to True.
        seed (int, optional): random seed for reproducibility. Defaults to 1738.
        n_iterations_per_print (int, optional): number of training iterations before printing loss. Defaults to 50.

    Returns:
        dict: a dictionary containing training and test losses
    """
    torch.manual_seed(seed)
    
    n_samples = X_train.shape[0]
    if n_samples > 10_000:
        warnings.warn(f'Dataset is large ({n_samples} samples). Training may be slow - consider a different training implementation that uses batches.')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # they used LBFGS in the paper but sticking to ADAM for now

    train_mses = torch.zeros(size=(num_itrs,))
    test_mses = torch.zeros(size=(num_itrs,))

    train_r2s = torch.zeros(size=(num_itrs,))
    test_r2s = torch.zeros(size=(num_itrs,))

    train_maes = torch.zeros(size=(num_itrs,))
    test_maes = torch.zeros(size=(num_itrs,))

    train_rmses = torch.zeros(size=(num_itrs,))
    test_rmses = torch.zeros(size=(num_itrs,))

    for i in range(num_itrs):
        optimizer.zero_grad()

        y_hat_train = model(X_train)[:, 0]

        train_mse = F.mse_loss(y_hat_train, y_train)
        train_mae = F.l1_loss(y_hat_train.detach(), y_train)
        train_rmse = torch.sqrt(train_mse.detach())
        train_r2 = 1 - (torch.sum((y_train - y_hat_train.detach())**2) / torch.sum((y_train - torch.mean(y_train))**2))
        
        train_mse.backward()
        optimizer.step() # lbfgs requires closure?? 

        train_mses[i] = train_mse.detach()
        train_maes[i] = train_mae
        train_rmses[i] = train_rmse
        train_r2s[i] = train_r2

        # test
        y_hat_test = model(X_test).detach()[:, 0]

        test_mse = F.mse_loss(y_hat_test, y_test)
        test_mae = F.l1_loss(y_hat_test, y_test)
        test_rmse = torch.sqrt(test_mse)
        test_r2 = 1 - (torch.sum((y_test - y_hat_test.detach())**2) / torch.sum((y_test - torch.mean(y_test))**2))

        test_mses[i] = test_mse
        test_maes[i] = test_mae
        test_rmses[i] = test_rmse
        test_r2s[i] = test_r2

        if verbose and i % n_iterations_per_print == 0:
            print(f'Train iteration {i}, mse: {train_mse.item()}, r2: {train_r2.item()}, mae: {train_mae.item()}, rmse: {train_rmse.item()}')
            print(f'Test iteration {i}, mse: {test_mse.item()}, r2: {test_r2.item()}, mae: {test_mae.item()}, rmse: {test_rmse.item()}')
        
    results = {
        'train_mses': train_mses,
        'train_r2s': train_r2s,
        'train_maes': train_maes,
        'train_rmses': train_rmses,
        'test_mses': test_mses,
        'test_r2s': test_r2s,
        'test_maes': test_maes,
        'test_rmses': test_rmses
    }

    return results
