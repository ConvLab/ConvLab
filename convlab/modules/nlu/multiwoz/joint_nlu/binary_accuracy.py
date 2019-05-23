# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional
from pprint import pprint

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("binary_accuracy")
class BinaryAccuracy(Metric):
    """
    This ``Metric`` calculates the binary accuracy.
    """
    def __init__(self) -> None:
        self._correct_count = 0.0
        self._total_count = 0.0

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ...).
        gold_labels : ``torch.Tensor``, required.
            A tensor of the same shape as ``predictions``.
        mask: ``torch.Tensor``, optional (default = None).
            A tensor of the same shape as ``predictions``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        absolute_errors = torch.abs(predictions - gold_labels)
        if mask is not None:
            absolute_errors *= mask
            total_count = torch.sum(mask)
        else:
            total_count = gold_labels.numel()
        error_count = torch.sum(absolute_errors)
        self._total_count += total_count
        self._correct_count += torch.sum(total_count - error_count)

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated mean absolute error.
        """
        accuracy = float(self._correct_count) / float(self._total_count)
        if reset:
            self.reset()
        return accuracy 

    @overrides
    def reset(self):
        self._correct_count = 0.0
        self._total_count = 0.0
