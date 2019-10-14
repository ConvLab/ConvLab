import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel


class BertNLU(nn.Module):
    def __init__(self, model_config, intent_dim, tag_dim, DEVICE, intent_weight=None):
        super(BertNLU, self).__init__()
        self.DEVICE = DEVICE
        self.bert = BertModel.from_pretrained(model_config['pre-trained'])
        for p in self.parameters():
            p.requires_grad = False
        self.intent_dim = intent_dim
        self.tag_dim = tag_dim
        self.dropout = nn.Dropout(model_config['dropout'])
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, self.intent_dim)
        self.tag_classifier = nn.Linear(self.bert.config.hidden_size, self.tag_dim)
        nn.init.xavier_uniform_(self.intent_classifier.weight)
        nn.init.xavier_uniform_(self.tag_classifier.weight)
        self.tag_loss = torch.nn.CrossEntropyLoss()
        self.intent_loss = torch.nn.BCEWithLogitsLoss(pos_weight=intent_weight)
        if model_config['optimizer'] == 'Adam':
            self.optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=model_config['lr'])
        else:
            self.optim = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=model_config['lr'])

    def forward(self, word_seq_tensor, word_mask_tensor):
        self.bert.eval()
        word_seq_tensor = word_seq_tensor.to(self.DEVICE)
        word_mask_tensor = word_mask_tensor.to(self.DEVICE)
        with torch.no_grad():
            encoder_layers, pooled_output = self.bert(input_ids=word_seq_tensor,
                                                      attention_mask=word_mask_tensor,
                                                      output_all_encoded_layers=False)
            # encoder_layers = [batch_size, sequence_length, hidden_size]
            # pooled_output = [batch_size, hidden_size]
        encoder_layers = self.dropout(encoder_layers)
        pooled_output = self.dropout(pooled_output)
        tag_logits = self.tag_classifier(encoder_layers)
        intent_logits = self.intent_classifier(pooled_output)
        # tag_logits = [batch_size, sequence_length, tag_dim]
        # intent_logits = [batch_size, intent_dim]
        return intent_logits, tag_logits

    def train_batch(self, word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor):
        self.train()
        self.optim.zero_grad()

        word_seq_tensor = word_seq_tensor.to(self.DEVICE)
        tag_seq_tensor = tag_seq_tensor.to(self.DEVICE)
        intent_tensor = intent_tensor.to(self.DEVICE, torch.float)
        word_mask_tensor = word_mask_tensor.to(self.DEVICE)
        tag_mask_tensor = tag_mask_tensor.to(self.DEVICE)

        intent_logits, tag_logits = self.forward(word_seq_tensor, word_mask_tensor)

        active_tag_loss = tag_mask_tensor.view(-1) == 1
        active_tag_logits = tag_logits.view(-1, self.tag_dim)[active_tag_loss]
        active_tag_labels = tag_seq_tensor.view(-1)[active_tag_loss]
        intent_loss = self.intent_loss(intent_logits, intent_tensor)
        tag_loss = self.tag_loss(active_tag_logits, active_tag_labels)
        loss = intent_loss + tag_loss
        loss.backward()
        self.optim.step()
        return intent_loss.item(), tag_loss.item(), loss.item(), intent_logits, tag_logits

    def eval_batch(self, word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor):
        self.eval()
        with torch.no_grad():
            word_seq_tensor = word_seq_tensor.to(self.DEVICE)
            tag_seq_tensor = tag_seq_tensor.to(self.DEVICE)
            intent_tensor = intent_tensor.to(self.DEVICE, torch.float)
            word_mask_tensor = word_mask_tensor.to(self.DEVICE)
            tag_mask_tensor = tag_mask_tensor.to(self.DEVICE)

            intent_logits, tag_logits = self.forward(word_seq_tensor, word_mask_tensor)

            active_tag_loss = tag_mask_tensor.view(-1) == 1
            active_tag_logits = tag_logits.view(-1, self.tag_dim)[active_tag_loss]
            active_tag_labels = tag_seq_tensor.view(-1)[active_tag_loss]
            intent_loss = self.intent_loss(intent_logits, intent_tensor)
            tag_loss = self.tag_loss(active_tag_logits, active_tag_labels)
            loss = intent_loss + tag_loss
        return intent_loss.item(), tag_loss.item(), loss.item(), intent_logits, tag_logits

    def predict_batch(self, word_seq_tensor, word_mask_tensor):
        self.eval()
        with torch.no_grad():
            intent_logits, tag_logits = self.forward(word_seq_tensor, word_mask_tensor)
        return intent_logits, tag_logits
