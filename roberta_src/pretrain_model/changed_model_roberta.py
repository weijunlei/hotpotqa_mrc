import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import math

from modeling_roberta import BertPreTrainedModel, RobertaModel, BertOutput, BertSelfOutput, BertIntermediate
from transformer import TransformerLayer
from transformers.configuration_roberta import RobertaConfig

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    'distilroberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    'roberta-base-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    'roberta-large-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}


class RobertaForParaClassification(BertPreTrainedModel):

    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForParaClassification, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, cls_mask=None, cls_label=None,cls_weight=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            token_type_ids=token_type_ids.unsqueeze(0)
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output).squeeze(-1)
        if cls_label is not None:
            # loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
            loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
            logits=logits*cls_mask.float()
            # loss = loss_fn(logits, cls_label.float())
            loss1 = loss_fn1(logits, cls_label.float())
            # weighted_loss=(loss*cls_mask.float())*cls_weight
            weighted_loss1 = (loss1 * cls_mask.float()) * cls_weight
            # loss=torch.sum(weighted_loss,(-1,-2),keepdim=False)
            loss1 = torch.sum(weighted_loss1, (-1, -2), keepdim=False)
            logits=torch.sigmoid(logits)
            # print(loss,loss1)
            return loss1, logits  # (loss), scores, (hidden_states), (attentions)
        else:
            return logits


class RobertaForQuestionAnsweringForwardBest(BertPreTrainedModel):
    def __init__(self, config):
        super(RobertaForQuestionAnsweringForwardBest, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                entity_ids=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape)<2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        sequence_output = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        sequence_output = self.dropout(sequence_output)
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask #*context_mask.float()
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask #*context_mask.float()
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        # sent_logits = self.sent(sequence_output).squeeze(-1)
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
            # sent_loss = (sent_loss * sent_mask.float())
            sent_loss = torch.sum(sent_loss, (-1, -2), keepdim=False)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions) # bsz*seq bsz*n
            end_loss = loss_fct(end_logits, end_positions)
            ans_loss = start_loss + end_loss
            total_loss = ans_loss + 0.2 * sent_loss
            return total_loss, start_logits, end_logits, sent_logits
        else:
            start_logits = nn.Softmax(dim=-1)(start_logits)
            end_logits = nn.Softmax(dim=-1)(end_logits)
            sent_logits = torch.sigmoid(sent_logits)
            return start_logits, end_logits, sent_logits


class RobertaForQuestionAnsweringForwardWithEntity(BertPreTrainedModel):
    def __init__(self, config):
        super(RobertaForQuestionAnsweringForwardWithEntity, self).__init__(config)
        self.roberta = RobertaModel(config)
        ENTITY_NUM = 20
        ENTITY_DIM = 40
        self.entity_embedder = nn.Embedding(num_embeddings=ENTITY_NUM, embedding_dim=ENTITY_DIM)
        self.start_logits = nn.Linear(config.hidden_size + ENTITY_DIM, 1)
        self.end_logits = nn.Linear(config.hidden_size + ENTITY_DIM, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size + ENTITY_DIM, 1)
        self.init_weights()

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                entity_ids=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        sequence_output = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        sequence_output = self.dropout(sequence_output)
        entity_output = self.entity_embedder(entity_ids)
        sequence_output = torch.cat([sequence_output, entity_output], dim=-1)
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask #*context_mask.float()
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask #*context_mask.float()
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        # sent_logits = self.sent(sequence_output).squeeze(-1)
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
            # sent_loss = (sent_loss * sent_mask.float())
            sent_loss = torch.sum(sent_loss, (-1, -2), keepdim=False)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)  # bsz * seq bsz*n
            end_loss = loss_fct(end_logits, end_positions)
            ans_loss = start_loss + end_loss
            total_loss = ans_loss + 0.2 * sent_loss
            return total_loss, start_logits, end_logits, sent_logits
        else:
            start_logits = nn.Softmax(dim=-1)(start_logits)
            end_logits = nn.Softmax(dim=-1)(end_logits)
            sent_logits = torch.sigmoid(sent_logits)
            return start_logits, end_logits, sent_logits