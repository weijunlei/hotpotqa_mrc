import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import math

from transformers import ElectraPreTrainedModel, ElectraModel
from modeling_bert import BertPreTrainedModel, BertModel


class ElectraForParagraphClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraForParagraphClassification, self).__init__(config)

        self.bert = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, cls_mask=None, cls_label=None, cls_weight=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        cls_output = outputs[1]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output).squeeze(-1)
        if cls_label is None:
            return logits
        # 选择第一个标记结果
        cls_label = torch.index_select(cls_label, dim=1, index=torch.tensor([0, ]).cuda())
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.config.num_labels), cls_label.view(-1))
        return loss, logits  # (loss), scores, (hidden_states), (attentions)


class ElectraForRelatedSentence(ElectraPreTrainedModel):

    def __init__(self, config):
        super(ElectraForRelatedSentence, self).__init__(config)

        self.bert = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, cls_mask=None, cls_label=None, cls_weight=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
        sequence_output = self.bert(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       head_mask=head_mask)
        sequence_output = sequence_output[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output).squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        logits = logits * cls_mask.float()
        if cls_label is None:
            return logits
        loss1 = loss_fn1(logits, cls_label.float())
        weighted_loss1 = (loss1 * cls_mask.float()) * cls_weight
        loss1 = torch.sum(weighted_loss1, (-1, -2), keepdim=False)
        logits = torch.sigmoid(logits)
        return loss1, logits

