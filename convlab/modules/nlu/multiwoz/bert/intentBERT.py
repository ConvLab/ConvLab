import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel


class IntentBERT(BertPreTrainedModel):
    def __init__(self, config, device, intent_dim, intent_weight):
        super(IntentBERT, self).__init__(config)
        self.num_labels = intent_dim
        self.device = device
        self.intent_weight = intent_weight

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.intent_weight)

        self.init_weights()

    def forward(self, word_seq_tensor, word_mask_tensor, intent_tensor):
        outputs = self.bert(input_ids=word_seq_tensor,
                            attention_mask=word_mask_tensor)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        intent_logits = self.classifier(pooled_output)

        outputs = (intent_logits,)

        if intent_tensor is not None:
            loss = self.loss_fct(intent_logits, intent_tensor)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits,
