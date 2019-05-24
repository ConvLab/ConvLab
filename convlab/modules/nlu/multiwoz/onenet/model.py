# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Optional, List, Any

from overrides import overrides
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
 
from convlab.modules.nlu.multiwoz.onenet.dai_f1_measure import DialogActItemF1Measure


@Model.register("onenet")
class OneNet(Model):
    """
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the tokens ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder that we will use in between embedding tokens and predicting output tags.
    sequence_label_namespace : ``str``, optional (default=``labels``)
        This is needed to compute the SpanBasedF1Measure metric.
        Unless you did something unusual, the default value should be what you want.
    feedforward : ``FeedForward``, optional, (default = None).
        An optional feedforward layer to apply after the encoder.
    label_encoding : ``str``, optional (default=``None``)
        Label encoding to use when calculating span f1 and constraining
        the CRF at decoding time . Valid options are "BIO", "BIOUL", "IOB1", "BMES".
        Required if ``calculate_span_f1`` or ``constrain_crf_decoding`` is true.
    include_start_end_transitions : ``bool``, optional (default=``True``)
        Whether to include start and end transition parameters in the CRF.
    constrain_crf_decoding : ``bool``, optional (default=``None``)
        If ``True``, the CRF is constrained at decoding time to
        produce valid sequences of tags. If this is ``True``, then
        ``label_encoding`` is required. If ``None`` and
        label_encoding is specified, this is set to ``True``.
        If ``None`` and label_encoding is not specified, it defaults
        to ``False``.
    calculate_span_f1 : ``bool``, optional (default=``None``)
        Calculate span-level F1 metrics during training. If this is ``True``, then
        ``label_encoding`` is required. If ``None`` and
        label_encoding is specified, this is set to ``True``.
        If ``None`` and label_encoding is not specified, it defaults
        to ``False``.
    dropout:  ``float``, optional (default=``None``)
    verbose_metrics : ``bool``, optional (default = False)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 tag_label_namespace: str = "labels",
                 domain_label_namespace: str = "domain_labels",
                 intent_label_namespace: str = "intent_labels",
                 feedforward: Optional[FeedForward] = None,
                 label_encoding: Optional[str] = None,
                 include_start_end_transitions: bool = True,
                 crf_decoding: bool = False,
                 constrain_crf_decoding: bool = None,
                 focal_loss_gamma: float = None,
                 calculate_span_f1: bool = None,
                 dropout: Optional[float] = None,
                 verbose_metrics: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.tag_label_namespace = tag_label_namespace
        self.intent_label_namespace = intent_label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size(tag_label_namespace)
        self.num_domains = self.vocab.get_vocab_size(domain_label_namespace)
        self.num_intents = self.vocab.get_vocab_size(intent_label_namespace)
        self.encoder = encoder
        self._verbose_metrics = verbose_metrics
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self._feedforward = feedforward

        # if feedforward is not None:
        #     output_dim = feedforward.get_output_dim()
        # else:
        #     output_dim = self.encoder.get_output_dim()
        self.tag_projection_layer = TimeDistributed(Linear(self.encoder.get_output_dim(),
                                                           self.num_tags))

        if self._feedforward is not None: 
            self.domain_projection_layer = Linear(feedforward.get_output_dim(), self.num_domains)
            self.intent_projection_layer = Linear(feedforward.get_output_dim(), self.num_intents)
        else:
            self.domain_projection_layer = Linear(self.encoder.get_output_dim(), self.num_domains)
            self.intent_projection_layer = Linear(self.encoder.get_output_dim(), self.num_intents)

        self.ce_loss = torch.nn.CrossEntropyLoss()

        # if  constrain_crf_decoding and calculate_span_f1 are not
        # provided, (i.e., they're None), set them to True
        # if label_encoding is provided and False if it isn't.
        if constrain_crf_decoding is None:
            constrain_crf_decoding = label_encoding is not None
        if calculate_span_f1 is None:
            calculate_span_f1 = label_encoding is not None

        self.label_encoding = label_encoding
        if constrain_crf_decoding:
            if not label_encoding:
                raise ConfigurationError("constrain_crf_decoding is True, but "
                                         "no label_encoding was specified.")
            labels = self.vocab.get_index_to_token_vocabulary(tag_label_namespace)
            constraints = allowed_transitions(label_encoding, labels)
        else:
            constraints = None

        self.include_start_end_transitions = include_start_end_transitions
        if crf_decoding:
            self.crf = ConditionalRandomField(
                    self.num_tags, constraints,
                    include_start_end_transitions=include_start_end_transitions
            )
        else:
            self.crf = None

        self._dai_f1_metric = DialogActItemF1Measure()
        # self.calculate_span_f1 = calculate_span_f1
        # if calculate_span_f1:
        #     if not label_encoding:
        #         raise ConfigurationError("calculate_span_f1 is True, but "
        #                                  "no label_encoding was specified.")
        #     self._f1_metric = SpanBasedF1Measure(vocab,
        #                                          tag_namespace=tag_label_namespace,
        #                                          label_encoding=label_encoding)

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        if feedforward is not None:
            check_dimensions_match(encoder.get_output_dim(), feedforward.get_input_dim(),
                                   "encoder output dim", "feedforward input dim")
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor = None,
                domain: torch.LongTensor = None,
                intent: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : ``Dict[str, torch.LongTensor]``, required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        tags : ``torch.LongTensor``, optional (default = ``None``)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containg the original words in the sentence to be tagged under a 'words' key.

        Returns
        -------
        An output dictionary consisting of:

        logits : ``torch.FloatTensor``
            The logits that are the output of the ``tag_projection_layer``
        mask : ``torch.LongTensor``
            The text field mask for the input tokens
        tags : ``List[List[int]]``
            The predicted tags using the Viterbi algorithm.
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised. Only computed if gold label ``tags`` are provided.
        """
        embedded_text_input = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)

        encoded_text = self.encoder(embedded_text_input, mask)

        if self.dropout:
            encoded_text = self.dropout(encoded_text)

        if self._feedforward is not None:
            encoded_summary = self._feedforward(util.get_final_encoder_states(
                encoded_text,
                mask,
                self.encoder.is_bidirectional()))
        else:
            encoded_summary = util.get_final_encoder_states(
                encoded_text,
                mask,
                self.encoder.is_bidirectional())

        tag_logits = self.tag_projection_layer(encoded_text)
        if self.crf:
            best_paths = self.crf.viterbi_tags(tag_logits, mask)
            # Just get the tags and ignore the score.
            predicted_tags = [x for x, y in best_paths]
        else:
            predicted_tags = self.get_predicted_tags(tag_logits)

        domain_logits = self.domain_projection_layer(encoded_summary)
        domain_probs = F.softmax(domain_logits, dim=-1)

        intent_logits = self.intent_projection_layer(encoded_summary)
        intent_probs = F.softmax(intent_logits, dim=-1)

        output = {"tag_logits": tag_logits, "mask": mask, "tags": predicted_tags,
        "domain_probs": domain_probs, "intent_probs": intent_probs}

        if tags is not None:
            if self.crf:
                # Add negative log-likelihood as loss
                log_likelihood = self.crf(tag_logits, tags, mask)
                output["loss"] = -log_likelihood

                # Represent viterbi tags as "class probabilities" that we can
                # feed into the metrics
                class_probabilities = tag_logits * 0.
                for i, instance_tags in enumerate(predicted_tags):
                    for j, tag_id in enumerate(instance_tags):
                        class_probabilities[i, j, tag_id] = 1
            else:
                loss = sequence_cross_entropy_with_logits(tag_logits, tags, mask)
                class_probabilities = tag_logits
                output["loss"] = loss

            # self.metrics['tag_acc'](class_probabilities, tags, mask.float())
            # if self.calculate_span_f1:
            #     self._f1_metric(class_probabilities, tags, mask.float())
        if domain is not None:
            output["loss"] += self.ce_loss(domain_logits, domain) 
        if intent is not None:
            output["loss"] += self.ce_loss(intent_logits, intent) 

        if metadata:
            output["words"] = [x["words"] for x in metadata]

        if tags is not None and metadata:
            self.decode(output)
            self._dai_f1_metric(output["dialog_act"], [x["dialog_act"] for x in metadata])

        return output


    def get_predicted_tags(self, sequence_logits: torch.Tensor) -> torch.Tensor:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = sequence_logits
        all_predictions = all_predictions.detach().cpu().numpy()
        if all_predictions.ndim == 3:
            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        for predictions in predictions_list:
            tags = np.argmax(predictions, axis=-1)
            all_tags.append(tags)
        return all_tags
 

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output_dict["tags"] = [
                [self.vocab.get_token_from_index(tag, namespace=self.tag_label_namespace)
                 for tag in instance_tags]
                for instance_tags in output_dict["tags"]
        ]

        argmax_indices = np.argmax(output_dict["domain_probs"].detach().cpu().numpy(), axis=-1)
        output_dict["domain"] = [self.vocab.get_token_from_index(x, namespace="domain_labels")
                       for x in argmax_indices]

        argmax_indices = np.argmax(output_dict["intent_probs"].detach().cpu().numpy(), axis=-1)
        output_dict["intent"] = [self.vocab.get_token_from_index(x, namespace="intent_labels")
                       for x in argmax_indices]

        output_dict["dialog_act"] = [] 
        for i, domain in enumerate(output_dict["domain"]):
            if "+" not in output_dict["intent"][i]:
                tags = []
                seq_len = len(output_dict["words"][i])
                for span in bio_tags_to_spans(output_dict["tags"][i][:seq_len]):
                    tags.append([span[0], " ".join(output_dict["words"][i][span[1][0]: span[1][1]+1])])
                intent = output_dict["intent"][i] if len(tags) > 0 else "None"
            else:
                intent, value = output_dict["intent"][i].split("*", 1)
                intent, slot = intent.split("+")
                tags = [[slot, value]]
            #     tags.append([output_dict["intent"][i].split("+")[1], "?"])
            # if len(tags) == 0:
            #     tags = [["none", "none"]]
            dialog_act = {domain+"-"+intent: tags} if domain != "None" and intent != "None" else {}
            output_dict["dialog_act"].append(dialog_act)

        return output_dict


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._dai_f1_metric.get_metric(reset=reset)
