import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
import warnings

BINARY_CLASSIFICATION_METRICS = [
    'train_cross_entropy',
    'test_cross_entropy',
    'train_accuracy',
    'test_accuracy',
    'train_precision',
    'test_precision',
    'train_recall',
    'test_recall',
    'train_f1',
    'test_f1'
]

def train_binary_classification(model: torch.nn.Module, X_train: torch.tensor , y_train: torch.tensor, X_test: torch.tensor, y_test: torch.tensor, 
                     lr: float = 0.01, num_itrs: int = 500, verbose: bool = True, seed: int = 1738, n_iterations_per_print: int = 50, data_checks: bool = True) -> dict:
    """Train a Kolmogorov Arnold Network (KAN) to minimize cross entropy loss for binary classification.

    Expects the model to return logits rather than probabilities.

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
        data_checks (bool, optional): whether to perform data checks on model inputs and outputs. Defaults to True.

    Returns:
        dict: a dictionary containing training and test losses
    """
    torch.manual_seed(seed)
    
    n_samples = X_train.shape[0]

    if data_checks:
        if n_samples > 10_000:
            warnings.warn(f'Dataset is large ({n_samples} samples). Training may be slow - consider a different training implementation that uses batches.')

        # this is binary classification...
        if not torch.all(torch.logical_or(y_train == 0, y_train == 1)):
            raise ValueError('y_train must be binary (0 or 1).')

        # are they all the same class?
        if torch.all(y_train == 0) or torch.all(y_train == 1):
            raise ValueError('y_train must not be all the same class (0 or 1).')


    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # they used LBFGS in the paper but sticking to ADAM for now

    loss_fn = torch.nn.BCEWithLogitsLoss()

    results = {metric: torch.zeros(size=(num_itrs,)) for metric in BINARY_CLASSIFICATION_METRICS}

    for i in range(num_itrs):
        optimizer.zero_grad()

        y_hat_train = model(X_train)[:, 0]

        if data_checks and i == 0:
            # some safety checks
            if torch.any(torch.isnan(y_hat_train)):
                raise ValueError('NaN values in y_hat_train. Check your model.')
            
            if torch.any(torch.isinf(y_hat_train)):
                raise ValueError('Inf values in y_hat_train. Check your model.')
            
        # give the model a second to learn before checking whether its outputs are logits
        if data_checks and i == 100 or i == num_itrs - 1:
            # are they all between 0 and 1? If so they might not be logits
            if torch.all(y_hat_train >= 0) and torch.all(y_hat_train <= 1):
                warnings.warn('y_hat_train values are between 0 and 1. Is your model returning logits? This function expects models to return logits, not probabilities.')

        train_cross_entropy = loss_fn(y_hat_train, y_train)
        
        train_cross_entropy.backward()
        optimizer.step() # lbfgs requires closure?? 

        results['train_cross_entropy'][i] = train_cross_entropy.detach()

        predictions = (F.sigmoid(y_hat_train) > 0.5).float().detach()

        clf_report = classification_report(y_train.detach(), predictions, output_dict=True, zero_division=0) # this will return 0 when there is an NaN/Inf value. So don't think there's a problem if all metrics are 0 on the first iteration
        clf_report_positive = clf_report['1.0']

        results['train_accuracy'][i] = clf_report['accuracy']
        results['train_precision'][i] = clf_report_positive['precision']
        results['train_recall'][i] = clf_report_positive['recall']
        results['train_f1'][i] = clf_report_positive['f1-score']

        with torch.no_grad():
            # test
            y_hat_test = model(X_test).detach()[:, 0]

            test_cross_entropy = loss_fn(y_hat_test, y_test)

            results['test_cross_entropy'][i] = test_cross_entropy

            predictions = (F.sigmoid(y_hat_test) > 0.5).float().detach()

            clf_report = classification_report(y_test.detach(), predictions, output_dict=True, zero_division=0) # this will return 0 when there is an NaN/Inf value. So don't think there's a problem if all metrics are 0 on the first iteration
            clf_report_positive = clf_report['1.0']

            results['test_accuracy'][i] = clf_report['accuracy']
            results['test_precision'][i] = clf_report_positive['precision']
            results['test_recall'][i] = clf_report_positive['recall']
            results['test_f1'][i] = clf_report_positive['f1-score']

        if verbose and i % n_iterations_per_print == 0:
            print('-'*50, f'Iteration {i}', '-'*50)
            for metric in BINARY_CLASSIFICATION_METRICS:
                print(f'{metric}: {results[metric][i]}')

    return results
