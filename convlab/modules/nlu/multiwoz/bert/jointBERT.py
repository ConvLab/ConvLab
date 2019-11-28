import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel


class JointBERT(BertPreTrainedModel):
    def __init__(self, bert_config, model_config, device, slot_dim, intent_dim, intent_weight=None):
        super(JointBERT, self).__init__(bert_config)
        self.slot_num_labels = slot_dim
        self.intent_num_labels = intent_dim
        self.device = device
        self.intent_weight = intent_weight if intent_weight is not None else torch.tensor([1.]*intent_dim)

        self.bert = BertModel(bert_config)
        self.dropout = nn.Dropout(model_config['dropout'])
        self.context = model_config['context']
        self.finetune = model_config['finetune']
        self.context_grad = model_config['context_grad']
        if self.context:
            self.intent_classifier = nn.Linear(2 * bert_config.hidden_size, self.intent_num_labels)
            self.slot_classifier = nn.Linear(2 * bert_config.hidden_size, self.slot_num_labels)
            self.intent_hidden = nn.Linear(2 * bert_config.hidden_size, 2 * bert_config.hidden_size)
            self.slot_hidden = nn.Linear(2 * bert_config.hidden_size, 2 * bert_config.hidden_size)
        else:
            self.intent_classifier = nn.Linear(bert_config.hidden_size, self.intent_num_labels)
            self.slot_classifier = nn.Linear(bert_config.hidden_size, self.slot_num_labels)
            self.intent_hidden = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
            self.slot_hidden = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
        self.intent_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.intent_weight)
        self.slot_loss_fct = torch.nn.CrossEntropyLoss()

        self.init_weights()

    def forward(self, word_seq_tensor, word_mask_tensor, tag_seq_tensor=None, tag_mask_tensor=None,
                intent_tensor=None, context_seq_tensor=None, context_mask_tensor=None):
        if not self.finetune:
            self.bert.eval()
            with torch.no_grad():
                outputs = self.bert(input_ids=word_seq_tensor,
                                    attention_mask=word_mask_tensor)
        else:
            outputs = self.bert(input_ids=word_seq_tensor,
                                attention_mask=word_mask_tensor)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        if self.context and context_seq_tensor is not None:
            if not self.finetune or not self.context_grad:
                with torch.no_grad():
                    context_output = self.bert(input_ids=context_seq_tensor, attention_mask=context_mask_tensor)[1]
            else:
                context_output = self.bert(input_ids=context_seq_tensor, attention_mask=context_mask_tensor)[1]
            sequence_output = torch.cat(
                [context_output.unsqueeze(1).repeat(1, sequence_output.size(1), 1),
                 sequence_output], dim=-1)
            pooled_output = torch.cat([context_output, pooled_output], dim=-1)

        sequence_output = nn.functional.relu(self.dropout(self.slot_hidden(sequence_output)))
        pooled_output = nn.functional.relu(self.dropout(self.intent_hidden(pooled_output)))

        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)
        outputs = (slot_logits,)

        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)
        outputs = outputs + (intent_logits,)

        if tag_seq_tensor is not None:
            active_tag_loss = tag_mask_tensor.view(-1) == 1
            active_tag_logits = slot_logits.view(-1, self.slot_num_labels)[active_tag_loss]
            active_tag_labels = tag_seq_tensor.view(-1)[active_tag_loss]
            slot_loss = self.slot_loss_fct(active_tag_logits, active_tag_labels)

            outputs = outputs + (slot_loss,)

        if intent_tensor is not None:
            intent_loss = self.intent_loss_fct(intent_logits, intent_tensor)
            outputs = outputs + (intent_loss,)

        return outputs  # slot_logits, intent_logits, (slot_loss), (intent_loss),
