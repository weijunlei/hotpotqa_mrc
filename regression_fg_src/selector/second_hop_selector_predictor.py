from __future__ import absolute_import, division, print_function
import os
import json

import argparse
import collections
import json
import logging
import math
import os
import random
import sys
from io import open
from pathlib import Path
import re
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Sampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer, RobertaTokenizer, ElectraTokenizer, AlbertTokenizer
from tqdm import tqdm, trange
from second_hop_prediction_helper import prediction_evaluate

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from collections import Counter
import string
import gc

from second_hop_data_helper import (HotpotQAExample,
                                    HotpotInputFeatures,
                                    read_second_hotpotqa_examples,
                                    convert_examples_to_second_features)
from second_selector_predictor_config import get_config

sys.path.append("../pretrain_model")
from changed_model import BertForParagraphClassification, BertForRelatedSentence, \
    ElectraForParagraphClassification, ElectraForRelatedSentence, \
    RobertaForParagraphClassification, RobertaForRelatedSentence, \
    BertForParagraphClassificationMean, BertForParagraphClassificationMax, \
    ElectraForParagraphClassificationCrossAttention, ElectraSecondForParagraphClassificationCrossAttention, \
    ElectraForParagraphClassificationRegression, ElectraForParagraphClassificationTwoRegression
from optimization import BertAdam, warmup_linear

models_dict = {"BertForRelatedSentence": BertForRelatedSentence,
               "BertForParagraphClassification": BertForParagraphClassification,
               "BertForParagraphClassificationMean": BertForParagraphClassificationMean,
               "BertForParagraphClassificationMax": BertForParagraphClassificationMax,
               "ElectraForParagraphClassification": ElectraForParagraphClassification,
               "ElectraForRelatedSentence": ElectraForRelatedSentence,
               "RobertaForParagraphClassification": RobertaForParagraphClassification,
               "RobertaForRelatedSentence": RobertaForRelatedSentence,
               "ElectraForParagraphClassificationCrossAttention": ElectraForParagraphClassificationCrossAttention,
               "ElectraSecondForParagraphClassificationCrossAttention": ElectraSecondForParagraphClassificationCrossAttention,
               "ElectraForParagraphClassificationRegression": ElectraForParagraphClassificationRegression,
               "ElectraForParagraphClassificationTwoRegression": ElectraForParagraphClassificationTwoRegression,
               }

# 日志设置
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def write_second_predict_result(args, examples, features, results, has_sentence_result=True):
    """ 给出预测结果 """
    paragraph_results = {}
    labels = {}
    dev_data = json.load(open(args.dev_file))
    for info in dev_data:
        context = info['context']
        get_id = info['_id']
        if len(context) <= 2:
            paragraph_results[get_id] = list(range(len(context)))
        supporting_facts = info['supporting_facts']
        supporting_facts_dict = set(['{}${}'.format(x[0], x[1]) for x in supporting_facts])
        for idx, paragraph in enumerate(context):
            title, sentences = paragraph
            related = False
            for sent_idx, sent in enumerate(sentences):
                if '{}${}'.format(title, sent_idx) in supporting_facts_dict:
                    related = True
            if related:
                if get_id not in labels:
                    labels[get_id] = []
                labels[get_id].append(idx)
    dev_related_paragraph_dict = json.load(
        open("{}/{}".format(args.first_predict_result_path, args.related_paragraph_file), "r"))
    for p_result in results:
        unique_id = p_result.unique_id
        qas_id = unique_id.split("_")[0]
        logit = p_result.logit
        min_num = min(logit)
        min_idx = logit.index(min_num)
        get_related_paragraphs = dev_related_paragraph_dict[qas_id]
        get_related_paragraphs = sorted(get_related_paragraphs)
        predict_result = []
        for idx, paragraph_num in enumerate(get_related_paragraphs):
            if idx == min_idx:
                continue
            predict_result.append(paragraph_num)
        paragraph_results[qas_id] = predict_result
    return paragraph_results


def run_predict(args):
    """ 预测结果 """
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    # preprocess_data
    # 模型和分词器配置
    cls_token = '[CLS]'
    sep_token = '[SEP]'
    unk_token = '[UNK]'
    pad_token = '[PAD]'
    if 'electra' in args.bert_model.lower():
        if not args.no_network:
            tokenizer = ElectraTokenizer.from_pretrained(args.bert_model,
                                                         do_lower_case=args.do_lower_case)
        else:
            tokenizer = ElectraTokenizer.from_pretrained(args.checkpoint_path,
                                                         do_lower_case=args.do_lower_case)
    elif 'albert' in args.bert_model.lower():
        cls_token = '[CLS]'
        sep_token = '[SEP]'
        pad_token = '<pad>'
        unk_token = '<unk>'
        if not args.no_network:
            tokenizer = AlbertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        else:
            tokenizer = AlbertTokenizer.from_pretrained(args.checkpoint_path, do_lower_case=args.do_lower_case)

    elif 'roberta' in args.bert_model.lower():
        cls_token = '<s>'
        sep_token = '</s>'
        unk_token = '<unk>'
        pad_token = '<pad>'
        if not args.no_network:
            tokenizer = RobertaTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        else:
            tokenizer = RobertaTokenizer.from_pretrained(args.checkpoint_path, do_lower_case=args.do_lower_case)
    elif 'bert' in args.bert_model.lower():
        if not args.no_network:
            tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        else:
            tokenizer = BertTokenizer.from_pretrained(args.checkpoint_path, do_lower_case=args.do_lower_case)
    else:
        raise ValueError("Not implement!")

    # 从文件中加载模型
    model = models_dict[args.model_name].from_pretrained(args.checkpoint_path)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    dev_examples = read_second_hotpotqa_examples(args=args,
                                                 input_file=args.dev_file,
                                                 related_paragraph_file="{}/{}".format(args.first_predict_result_path,
                                                                                       args.related_paragraph_file),

                                                 is_training='test')
    example_num = len(dev_examples)
    max_train_data_size = 100000
    start_idxs = list(range(0, example_num, max_train_data_size))
    end_idxs = [x + max_train_data_size for x in start_idxs]
    end_idxs[-1] = example_num
    all_results = []
    related_paragraph = {}
    total = 0
    max_len = 0

    data = json.load(open(args.dev_file, 'r', encoding='utf-8'))
    first_best_paragraph_file = "{}/{}".format(args.first_predict_result_path, args.best_paragraph_file)
    first_best_paragraph = json.load(open(first_best_paragraph_file, 'r', encoding='utf-8'))

    for start_idx in range(len(start_idxs)):
        logger.info("start idx: {} all idx length: {}".format(start_idx, len(start_idxs)))
        truly_examples = dev_examples[start_idxs[start_idx]: end_idxs[start_idx]]
        truly_features = convert_examples_to_second_features(
            examples=truly_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            is_training='test',
            cls_token=cls_token,
            sep_token=sep_token,
            unk_token=unk_token,
            pad_token=pad_token
        )
        logger.info("all truly gotten features: {}".format(len(truly_features)))
        d_all_input_ids = torch.tensor([f.input_ids for f in truly_features], dtype=torch.long)
        d_all_input_mask = torch.tensor([f.input_mask for f in truly_features], dtype=torch.long)
        d_all_segment_ids = torch.tensor([f.segment_ids for f in truly_features], dtype=torch.long)
        d_all_cls_mask = torch.tensor([f.cls_mask for f in truly_features], dtype=torch.long)
        d_pq_end_pos = torch.tensor([f.pq_end_pos for f in truly_features], dtype=torch.long)
        d_all_cls_weight = torch.tensor([f.cls_weight for f in truly_features], dtype=torch.float)
        d_all_example_index = torch.arange(d_all_input_ids.size(0), dtype=torch.long)
        dev_data = TensorDataset(d_all_input_ids, d_all_input_mask, d_all_segment_ids,
                                 d_all_cls_mask, d_pq_end_pos, d_all_cls_weight, d_all_example_index)
        if args.local_rank == -1:
            dev_sampler = SequentialSampler(dev_data)
        else:
            dev_sampler = DistributedSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.val_batch_size)
        RawResult = collections.namedtuple("RawResult",
                                           ["unique_id", "logit"])

        model.eval()
        has_sentence_result = True
        if args.model_name == 'BertForParagraphClassification' or 'ParagraphClassification' in args.model_name:
            has_sentence_result = False
        with torch.no_grad():
            cur_result = []
            for batch_idx, batch in enumerate(
                    tqdm(dev_dataloader, desc="predict interation: {}".format(args.dev_file))):
                # example index getter
                d_example_indices = batch[-1]
                # 多gpu训练的scatter
                if n_gpu == 1:
                    # 去除example index
                    batch = tuple(x.squeeze(0).to(device) for x in batch[:-1])
                else:
                    batch = batch[:-1]
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "cls_mask": batch[3],
                    "pq_end_pos": batch[4],
                    "cls_weight": batch[5]
                }
                # 获取预测结果
                dev_logits = model(**inputs)
                dev_logits = torch.sigmoid(dev_logits)
                for i, example_index in enumerate(d_example_indices):
                    # start_position = start_positions[i].detach().cpu().tolist()
                    # end_position = end_positions[i].detach().cpu().tolist()
                    if not has_sentence_result:
                        dev_logit = dev_logits[i].detach().cpu().tolist()
                        dev_cls_mask = inputs["cls_mask"][i].detach().cpu().tolist()
                        predict_paragraph_result = []
                        for logit_num, mask_num in zip(dev_logit, dev_cls_mask):
                            if mask_num == 1:
                                predict_paragraph_result.append(logit_num)
                        dev_logit = predict_paragraph_result
                    else:
                        dev_logit = dev_logits[i].detach().cpu().tolist()
                    dev_feature = truly_features[example_index.item()]
                    unique_id = dev_feature.unique_id
                    cur_result.append(RawResult(unique_id=unique_id,
                                                logit=dev_logit))
            tmp_paragraph_results = write_second_predict_result(
                args=args,
                examples=truly_examples,
                features=truly_features,
                results=cur_result,
                has_sentence_result=has_sentence_result
            )
            all_results += cur_result
            related_paragraph.update(tmp_paragraph_results)
            del tmp_paragraph_results
            del d_example_indices, inputs
            gc.collect()
        del d_all_input_ids, d_all_input_mask, d_all_segment_ids, d_all_cls_mask, d_all_cls_weight, d_all_example_index, d_pq_end_pos
        del truly_examples, truly_features, dev_data
        gc.collect()
    logger.info("writing result to file...")
    paragraph_results = {}
    labels = {}
    dev_data = json.load(open(args.dev_file))
    for info in dev_data:
        context = info['context']
        get_id = info['_id']
        if len(context) <= 2:
            paragraph_results[get_id] = list(range(len(context)))
        supporting_facts = info['supporting_facts']
        supporting_facts_dict = set(['{}${}'.format(x[0], x[1]) for x in supporting_facts])
        for idx, paragraph in enumerate(context):
            title, sentences = paragraph
            related = False
            for sent_idx, sent in enumerate(sentences):
                if '{}${}'.format(title, sent_idx) in supporting_facts_dict:
                    related = True
            if related:
                if get_id not in labels:
                    labels[get_id] = []
                labels[get_id].append(idx)

    dev_related_paragraph_dict = json.load(
        open("{}/{}".format(args.first_predict_result_path, args.related_paragraph_file), "r"))
    for p_result in all_results:
        unique_id = p_result.unique_id
        qas_id = unique_id.split("_")[0]
        logit = p_result.logit
        min_num = min(logit)
        min_idx = logit.index(min_num)
        get_related_paragraphs = dev_related_paragraph_dict[qas_id]
        get_related_paragraphs = sorted(get_related_paragraphs)
        predict_result = []
        for idx, paragraph_num in enumerate(get_related_paragraphs):
            if idx == min_idx:
                continue
            predict_result.append(paragraph_num)
        paragraph_results[qas_id] = predict_result
    if not os.path.exists(args.second_predict_result_path):
        logger.info("make new output dir:{}".format(args.second_predict_result_path))
        os.makedirs(args.second_predict_result_path)
    final_related_paragraph_file = "{}/{}".format(args.second_predict_result_path, args.final_related_result)
    json.dump(paragraph_results, open(final_related_paragraph_file, "w", encoding='utf-8'))
    logger.info("write result done!")
    acc, acc, acc, em = prediction_evaluate(args,
                                            paragraph_results=paragraph_results,
                                            labels=labels,
                                            step=0)
    print("acc: {} em: {}".format(acc, em))


if __name__ == '__main__':
    parser = get_config()
    args = parser.parse_args()
    run_predict(args=args)
