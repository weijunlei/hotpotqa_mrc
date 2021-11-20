import pdb

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import RobertaModel


class RobertaForParagraphClassification(RobertaModel):
    def __init__(self, config):
        super(RobertaForParagraphClassification, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                label=None,
                ):
        if len(input_ids) == 1:
            input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            if label is not None:
                label = label.unsqueeze(0)
        outputs = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        cls_output = outputs[1]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        if label is None:
            return logits
        else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, label)
            return loss, logits
