import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel


class SlotBERT(BertPreTrainedModel):
    def __init__(self, config, device, slot_dim):
        super(SlotBERT, self).__init__(config)
        self.num_labels = slot_dim
        self.device = device

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss()

        self.init_weights()

    def forward(self, word_seq_tensor, word_mask_tensor, tag_seq_tensor, tag_mask_tensor):
        outputs = self.bert(input_ids=word_seq_tensor,
                            attention_mask=word_mask_tensor)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        slot_logits = self.classifier(sequence_output)

        outputs = (slot_logits,)

        if tag_seq_tensor is not None:
            active_tag_loss = tag_mask_tensor.view(-1) == 1
            active_tag_logits = slot_logits.view(-1, self.num_labels)[active_tag_loss]
            active_tag_labels = tag_seq_tensor.view(-1)[active_tag_loss]
            loss = self.loss_fct(active_tag_logits, active_tag_labels)

            outputs = (loss,) + outputs

        return outputs  # (loss), logits,
