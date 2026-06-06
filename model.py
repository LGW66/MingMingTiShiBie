import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BertNER(nn.Module):
    def __init__(self, num_labels, model_name='bert-base-chinese'):
        super(BertNER, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        
        return loss, logits
    
    def predict(self, input_ids, attention_mask, token_type_ids=None):
        _, logits = self.forward(input_ids, attention_mask, token_type_ids)
        predictions = torch.argmax(logits, dim=-1)
        return predictions