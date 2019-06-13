from typing import Dict, Optional, List, Any

from overrides import overrides
import numpy as np
import torch
from torch.nn.modules.linear import Linear

from allennlp.data import Vocabulary
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("vanilla_mle_policy")
class VanillaMLE(Model):
    """
    The ``VanillaMLE`` makes predictions based on a Softmax over a list of top combinatorial actions.

    Parameters
    ----------
    """

    def __init__(self, vocab: Vocabulary,
                 input_dim: int,
                 num_classes: int,
                 label_namespace: str = "labels",
                 feedforward: Optional[FeedForward] = None,
                 dropout: Optional[float] = None,
                 verbose_metrics: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self.label_namespace = label_namespace
        self.input_dim = input_dim
        self.num_classes = num_classes 
        self._verbose_metrics = verbose_metrics
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self._feedforward = feedforward

        if self._feedforward is not None: 
            self.projection_layer = Linear(feedforward.get_output_dim(), self.num_classes)
        else:
            self.projection_layer = Linear(self.input_dim, self.num_classes)

        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3),
                "accuracy5": CategoricalAccuracy(top_k=5)
        }
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                states: torch.FloatTensor,
                actions: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------

        Returns
        -------
        """
        if self.dropout:
            states = self.dropout(states)

        if self._feedforward is not None:
            states = self._feedforward(states)

        logits = self.projection_layer(states)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        output = {"logits": logits, "probs": probs}

        if actions is not None:
            output["loss"] = self._loss(logits, actions)
            for metric in self.metrics.values():
                metric(logits, actions)
        
        return output

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities.
        """
        predictions = output_dict["probs"].detach().cpu().numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        output_dict["actions"] = argmax_indices 

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
