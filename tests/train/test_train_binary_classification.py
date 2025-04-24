import torch
import torch.nn.functional as F
from kan import KAN

from train import train_binary_classification
from train.train_binary_classification import BINARY_CLASSIFICATION_METRICS


class MetricThresholdEvaluator:
    def __init__(self, threshold_value, comparison):
        self.threshold_value = threshold_value
        self.comparison = comparison
    
    def evaluate(self, metric_value):
        if self.comparison == 'greater':
            return metric_value > self.threshold_value
        elif self.comparison == 'less':
            return metric_value < self.threshold_value
        elif self.comparison == 'equal':
            return metric_value == self.threshold_value
        else:
            raise ValueError(f"Invalid comparison: {self.comparison}. Must be 'greater', 'less', or 'equal'.")


BINARY_CLASSIFICATION_METRIC_THRESHOLDS = {
    'train_cross_entropy': MetricThresholdEvaluator(threshold_value=0.01, comparison='less'),
    'train_accuracy': MetricThresholdEvaluator(threshold_value=0.95, comparison='greater'),
    'train_precision': MetricThresholdEvaluator(threshold_value=0.95, comparison='greater'),
    'train_recall': MetricThresholdEvaluator(threshold_value=0.95, comparison='greater'),
    'train_f1': MetricThresholdEvaluator(threshold_value=0.95, comparison='greater'),
    'test_cross_entropy': MetricThresholdEvaluator(threshold_value=0.01, comparison='less'),
    'test_accuracy': MetricThresholdEvaluator(threshold_value=0.95, comparison='greater'),
    'test_precision': MetricThresholdEvaluator(threshold_value=0.95, comparison='greater'),
    'test_recall': MetricThresholdEvaluator(threshold_value=0.95, comparison='greater'),
    'test_f1': MetricThresholdEvaluator(threshold_value=0.95, comparison='greater'),
}


def test_metric_thresholds():
    # make sure all the metrics have associated thresholds
    for metric in BINARY_CLASSIFICATION_METRICS:
        assert metric in BINARY_CLASSIFICATION_METRIC_THRESHOLDS, f'{metric} not in BINARY_CLASSIFICATION_METRIC_THRESHOLDS'


def test_train_binary_classification():
    """
    Try to fit a KAN to target values produced by a randomly initialized model.
    """
    num_samples = 100
    num_features = 3
    num_itrs = 250

    torch.manual_seed(1738)
    dummy_model = KAN(width=[3, 2, 1])

    X = torch.rand(size=(num_samples, num_features))
    logits = dummy_model(X)[:, 0].detach()
    y = (logits > torch.median(logits)).to(torch.float32) # not really logits. I just want half to be positive class the other half negative class
    
    del dummy_model

    # train this model to match the randomly initialized dummy_model
    model = KAN(width=[3, 4, 4, 1])

    results = train_binary_classification(model, X_train=X, y_train=y, X_test=X, y_test=y, num_itrs=num_itrs, lr=0.01, verbose=False)

    # check the metrics are all there 
    for metric in BINARY_CLASSIFICATION_METRICS:
        assert metric in results, f'{metric} not in results'

        assert isinstance(results[metric], torch.Tensor), f'{metric} is not a tensor'
        assert results[metric].dtype == torch.float32, f'expcted {metric} to be dtype float32: instead got {results[metric].dtype}'
        assert results[metric].shape == (num_itrs,), f'{metric} shape is not correct: got {results[metric].shape} but expected ({num_itrs},)'
        assert torch.all(torch.isfinite(results[metric])), f'{metric} has non-finite values'
        assert torch.all(torch.isreal(results[metric])), f'{metric} has non-real values'
        assert not torch.any(torch.isnan(results[metric])), f'{metric} has NaN values'

        # check the metric meets the threshold
        threshold_evaluator = BINARY_CLASSIFICATION_METRIC_THRESHOLDS[metric]
        final_metric = results[metric][-1]
        assert threshold_evaluator.evaluate(final_metric), f"Final {metric} value {final_metric} is not {threshold_evaluator.comparison} than/to {threshold_evaluator.threshold_value}"

