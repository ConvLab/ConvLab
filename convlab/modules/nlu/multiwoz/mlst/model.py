# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Optional, List, Any

from overrides import overrides
import numpy as np
import torch
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

from convlab.modules.nlu.multiwoz.mlst.binary_accuracy import BinaryAccuracy
from convlab.modules.nlu.multiwoz.mlst.multilabel_f1_measure import MultiLabelF1Measure
from convlab.modules.nlu.multiwoz.mlst.focal_loss import FocalBCEWithLogitsLoss
from convlab.modules.nlu.multiwoz.mlst.dai_f1_measure import DialogActItemF1Measure


@Model.register("mlst_nlu")
class MlstNlu(Model):
    """
    The ``MlstNlu`` encodes a sequence of text with a ``Seq2SeqEncoder``,
    then uses a Conditional Random Field model to predict a tag for each token in the sequence.

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
                 intent_encoder: Seq2SeqEncoder = None,
                 sequence_label_namespace: str = "labels",
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

        self.sequence_label_namespace = sequence_label_namespace
        self.intent_label_namespace = intent_label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size(sequence_label_namespace)
        self.num_intents = self.vocab.get_vocab_size(intent_label_namespace)
        self.encoder = encoder
        self.intent_encoder = intent_encoder
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
            self.intent_projection_layer = Linear(feedforward.get_output_dim(), self.num_intents)
        else:
            self.intent_projection_layer = Linear(self.encoder.get_output_dim(), self.num_intents)

        if focal_loss_gamma is not None:
            self.intent_loss = FocalBCEWithLogitsLoss(gamma=focal_loss_gamma)
            # self.intent_loss2 = torch.nn.BCEWithLogitsLoss()
        else:
            self.intent_loss = torch.nn.BCEWithLogitsLoss()

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
            labels = self.vocab.get_index_to_token_vocabulary(sequence_label_namespace)
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

        # self.metrics = {
        #     "int_acc": BinaryAccuracy(),
        #     "tag_acc": CategoricalAccuracy()
        # }
        self._intent_f1_metric = MultiLabelF1Measure(vocab,
                                                namespace=intent_label_namespace)
        self.calculate_span_f1 = calculate_span_f1
        if calculate_span_f1:
            if not label_encoding:
                raise ConfigurationError("calculate_span_f1 is True, but "
                                          "no label_encoding was specified.")
            self._f1_metric = SpanBasedF1Measure(vocab,
                                                 tag_namespace=sequence_label_namespace,
                                                 label_encoding=label_encoding)
        self._dai_f1_metric = DialogActItemF1Measure()

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
                intents: torch.LongTensor = None,
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

        intent_encoded_text = self.intent_encoder(encoded_text, mask) if self.intent_encoder else encoded_text
        if self.dropout and self.intent_encoder:
            intent_encoded_text = self.dropout(intent_encoded_text)

        is_bidirectional = self.intent_encoder.is_bidirectional() if self.intent_encoder else self.encoder.is_bidirectional()
        if self._feedforward is not None:
            encoded_summary = self._feedforward(util.get_final_encoder_states(
                intent_encoded_text,
                mask,
                is_bidirectional))
        else:
            encoded_summary = util.get_final_encoder_states(
                intent_encoded_text,
                mask,
                is_bidirectional)

        sequence_logits = self.tag_projection_layer(encoded_text)
        if self.crf is not None:
            best_paths = self.crf.viterbi_tags(sequence_logits, mask)
            # Just get the tags and ignore the score.
            predicted_tags = [x for x, y in best_paths]
        else:
            predicted_tags = self.get_predicted_tags(sequence_logits)

        intent_logits = self.intent_projection_layer(encoded_summary)
        predicted_intents = (torch.sigmoid(intent_logits) > 0.5).long()

        output = {"sequence_logits": sequence_logits, "mask": mask, "tags": predicted_tags,
        "intent_logits": intent_logits, "intents": predicted_intents}

        if tags is not None:
            if self.crf is not None:
                # Add negative log-likelihood as loss
                log_likelihood = self.crf(sequence_logits, tags, mask)
                output["loss"] = -log_likelihood

                # Represent viterbi tags as "class probabilities" that we can
                # feed into the metrics
                class_probabilities = sequence_logits * 0.
                for i, instance_tags in enumerate(predicted_tags):
                    for j, tag_id in enumerate(instance_tags):
                        class_probabilities[i, j, tag_id] = 1
            else:
                loss = sequence_cross_entropy_with_logits(sequence_logits, tags, mask)
                class_probabilities = sequence_logits
                output["loss"] = loss

            # self.metrics['tag_acc'](class_probabilities, tags, mask.float())
            if self.calculate_span_f1:
                self._f1_metric(class_probabilities, tags, mask.float())
        
        if intents is not None:
            output["loss"] += self.intent_loss(intent_logits, intents.float()) 
            # bloss = self.intent_loss2(intent_logits, intents.float()) 

            # self.metrics['int_acc'](predicted_intents, intents)
            self._intent_f1_metric(predicted_intents, intents)

            # print(list([self.vocab.get_token_from_index(intent[0], namespace=self.intent_label_namespace) 
            # for intent in instance_intents.nonzero().tolist()] for instance_intents in predicted_intents))
            # print(list([self.vocab.get_token_from_index(intent[0], namespace=self.intent_label_namespace) 
            # for intent in instance_intents.nonzero().tolist()] for instance_intents in intents))

        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]

        if tags is not None and metadata:
            self.decode(output)
            # print(output)
            # print(metadata)
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
                [self.vocab.get_token_from_index(tag, namespace=self.sequence_label_namespace)
                 for tag in instance_tags]
                for instance_tags in output_dict["tags"]
        ]
        output_dict["intents"] = [
                [self.vocab.get_token_from_index(intent[0], namespace=self.intent_label_namespace) 
            for intent in instance_intents.nonzero().tolist()] 
            for instance_intents in output_dict["intents"]
        ]
        output_dict["dialog_act"] = []
        for i, tags in enumerate(output_dict["tags"]): 
            seq_len = len(output_dict["words"][i])
            spans = bio_tags_to_spans(tags[:seq_len])
            dialog_act = {}
            for span in spans:
                domain_act = span[0].split("+")[0]
                slot = span[0].split("+")[1]
                value = " ".join(output_dict["words"][i][span[1][0]:span[1][1]+1])
                if domain_act not in dialog_act:
                    dialog_act[domain_act] = [[slot, value]]
                else:
                    dialog_act[domain_act].append([slot, value])
            for intent in output_dict["intents"][i]:
                if "+" in intent: 
                    if "*" in intent: 
                        intent, value = intent.split("*", 1) 
                    else:
                        value = "?"
                    domain_act = intent.split("+")[0] 
                    if domain_act not in dialog_act:
                        dialog_act[domain_act] = [[intent.split("+")[1], value]]
                    else:
                        dialog_act[domain_act].append([intent.split("+")[1], value])
                else:
                    dialog_act[intent] = [["none", "none"]]
            output_dict["dialog_act"].append(dialog_act)

        return output_dict


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # return self._dai_f1_metric.get_metric(reset=reset)
        # metrics_to_return = {metric_name: metric.get_metric(reset) for
        #                      metric_name, metric in self.metrics.items()}

        metrics_to_return = {}
        intent_f1_dict = self._intent_f1_metric.get_metric(reset=reset)
        # if self._verbose_metrics:
        #     metrics_to_return.update(intent_f1_dict)
        # else:
        metrics_to_return.update({"int_"+x[:1]: y for x, y in intent_f1_dict.items() if "overall" in x})
        if self.calculate_span_f1:
            f1_dict = self._f1_metric.get_metric(reset=reset)
        #     if self._verbose_metrics:
        #         metrics_to_return.update(f1_dict)
        #     else:
            metrics_to_return.update({"tag_"+x[:1]: y for x, y in f1_dict.items() if "overall" in x})
        metrics_to_return.update(self._dai_f1_metric.get_metric(reset=reset))
        return metrics_to_return
