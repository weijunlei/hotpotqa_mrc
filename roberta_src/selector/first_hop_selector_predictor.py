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
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,Sampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import RobertaTokenizer
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from collections import Counter
import string
import gc

from first_predict_config import get_config
from first_hop_data_helper import (HotpotQAExample,
                                       HotpotInputFeatures,
                                       read_hotpotqa_examples,
                                       convert_examples_to_features)
from first_hop_selector import dev_feature_getter, write_predictions
from lazy_dataloader import LazyLoadTensorDataset
sys.path.append("../pretrain_model")
from changed_model_roberta import RobertaForParagraphClassification, RobertaForRelatedSentence
from optimization import BertAdam, warmup_linear

# 日志设置
logger = None


def logger_config(log_path, log_prefix='lwj', write2console=False):
    """
    日志配置
    :param log_path: 输出的日志路径
    :param log_prefix: 记录中的日志前缀
    :param write2console: 是否输出到命令行
    :return:
    """
    global logger
    logger = logging.getLogger(log_prefix)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if write2console:
        # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)
    # 为logger对象添加句柄
    logger.addHandler(handler)

    return logger


def write_predict_result(examples, features, results, has_sentence_result=True):
    """ 给出预测结果 """
    tmp_best_paragraph = {}
    tmp_related_sentence = {}
    tmp_related_paragraph = {}
    paragraph_results = {}
    sentence_results = {}
    example_index2features = collections.defaultdict(list)
    for feature in features:
        example_index2features[feature.example_index].append(feature)
    unique_id2result = {}
    for result in results:
        unique_id2result[result[0]] = result
    for example_idx, example in enumerate(examples):
        features = example_index2features[example_idx]
        # 将5a8b57f25542995d1e6f1371_0_0 qid_context_sent 切分为 qid_context
        id = '_'.join(features[0].unique_id.split('_')[:-1])
        sentence_result = []
        if len(features) == 1:
            # 对单实例单结果处理
            get_feature = features[0]
            get_feature_id = get_feature.unique_id
            # 对max_seq预测的结果
            raw_result = unique_id2result[get_feature_id].logit
            # 第一个'[CLS]'为paragraph为支撑句标识
            paragraph_results[id] = raw_result[0]
            labels_result = raw_result
            cls_masks = get_feature.cls_mask
            if has_sentence_result:
                for idx, (label_result, cls_mask) in enumerate(zip(labels_result, cls_masks)):
                    if idx == 0:
                        continue
                    if cls_mask != 0:
                        sentence_result.append(label_result)
                sentence_results[id] = sentence_result
                assert len(sentence_result) == sum(features[0].cls_mask) - 1
        else:
            # 对单实例的多结果处理
            paragraph_result = 0
            overlap = 0
            mask1 = 0
            roll_back = None
            for feature_idx, feature in enumerate(features):
                feature_result = unique_id2result[feature.unique_id].logit
                if feature_result[0] > paragraph_result:
                    paragraph_result = feature_result[0]
                if has_sentence_result:
                    tmp_sent_result = []
                    tmp_label_result = []
                    mask1 += sum(feature.cls_mask[1:])
                    label_results = feature_result[1:]
                    cls_masks = feature.cls_mask[1:]
                    for idx, (label_result, cls_mask) in enumerate(zip(label_results, cls_masks)):
                        if cls_mask != 0:
                            tmp_sent_result.append(label_result)
                    if roll_back is None:
                        roll_back = 0
                    elif roll_back == 1:
                        sentence_result[-1] = max(sentence_result[-1], tmp_sent_result[0])
                        tmp_sent_result = tmp_sent_result[1:]
                    elif roll_back == 2:
                        sentence_result[-2] = max(sentence_result[-2], tmp_sent_result[0])
                        sentence_result[-1] = max(sentence_result[-1], tmp_sent_result[1])
                        tmp_sent_result = tmp_sent_result[2:]
                    sentence_result += tmp_sent_result
                    overlap += roll_back
                    roll_back = feature.roll_back
            paragraph_results[id] = paragraph_result
            sentence_results[id] = sentence_result
            if has_sentence_result:
                assert len(sentence_result) + overlap == mask1
    context_dict = {}
    # 将每个段落的结果写入到context中
    for k, v in paragraph_results.items():
        context_id, paragraph_id = k.split('_')
        paragraph_id = int(paragraph_id)
        if context_id not in context_dict:
            context_dict[context_id] = [[0]*10, [0]*10]
        context_dict[context_id][0][paragraph_id] = v
    # 将context最大结果导出
    for k, v in context_dict.items():
        thread = 0.01
        max_v = max(v[0])
        min_v = min(v[0])
        max_logit = -1000
        max_result = False
        get_related_paras = []
        max_para = -1
        for idx, para_pro in enumerate(v[0]):
            if para_pro > max_logit and para_pro != 0:
                max_logit = para_pro
                max_para = idx
            para_pro = (para_pro - min_v) / (max_v - min_v)
            if para_pro > thread:
                get_related_paras.append(idx)
        tmp_best_paragraph[k] = max_para
        tmp_related_paragraph[k] = get_related_paras
    # 获取相关段落和句子
    if has_sentence_result:
        sentence_dict = {}
        for k, v in sentence_results.items():
            context_id, paragraph_id = k.split('_')
            paragraph_id = int(paragraph_id)
            if context_id not in sentence_dict:
                sentence_dict[context_id] = [[[]] * 10, [[]] * 10, [[]]*10]
            sentence_dict[context_id][0][paragraph_id] = v
        for k, v in sentence_dict.items():
            get_paragraph_idx = tmp_best_paragraph[k]
            pred_sent_result = v[0][get_paragraph_idx]
            real_sent_result = v[1][get_paragraph_idx]
            tmp_related_sentence[k] = [pred_sent_result, real_sent_result]
    return tmp_best_paragraph, tmp_related_sentence, tmp_related_paragraph


def run_predict(args, rank=0, world_size=2):
    """ 预测结果 """
    # 配置日志文件
    global logger
    if rank == 0 and not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    log_path = os.path.join(args.log_path, 'log_first_predictor_{}_{}_{}_{}'.format(
        args.log_prefix,
        args.bert_model.split('/')[-1],
        args.predict_result_path.split('/')[-1],
        args.max_seq_length,
    ))
    logger = logger_config(log_path=log_path, log_prefix='')
    logger.info('-' * 13 + '所有配置' + '-' * 13)
    logger.info("所有参数配置如下：")
    for arg in vars(args):
        logger.info('{}: {}'.format(arg, getattr(args, arg)))
    logger.info('-' * 30)
    # 多卡配置或训练配置如下
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    # preprocess_data
    tokenizer = RobertaTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    models_dict = {"RobertaForRelatedSentence": RobertaForRelatedSentence,
                   "RobertaForParagraphClassification": RobertaForParagraphClassification}
    # 从文件中加载模型
    model = models_dict[args.model_name].from_pretrained(args.checkpoint_path)

    if args.fp16:
        model.half()
    model.to(device)
    # if args.local_rank != -1:
    #     try:
    #         from apex.parallel import DistributedDataParallel as DDP
    #     except ImportError:
    #         raise ImportError(
    #             "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
    #
    #     model = DDP(model)
    # elif n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        logger.info("setting model {}..".format(rank))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
        logger.info("setting model {} done!".format(rank))
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    dev_examples = read_hotpotqa_examples(input_file=args.dev_file,
                                          is_training='test')
    example_num = len(dev_examples)
    logger.info("all examples number: {}".format(example_num))
    max_train_data_size = 100000
    start_idxs = list(range(0, example_num, max_train_data_size))
    end_idxs = [x + max_train_data_size for x in start_idxs]
    end_idxs[-1] = example_num
    all_results = []
    best_paragraph = {}
    related_sentence = {}
    related_paragraph = {}
    new_context = {}
    total = 0
    max_len = 0

    has_sentence_result = True

    if args.model_name == 'RobertaForParagraphClassification':
        has_sentence_result = False

    for idx in range(len(start_idxs)):
        logger.info("predict idx: {} all length: {}".format(idx, len(start_idxs)))
        truly_examples = dev_examples[start_idxs[idx]: end_idxs[idx]]
        truly_features = convert_examples_to_features(
            examples=truly_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            is_training='test'
        )
        logger.info("all truly gotten features: {}".format(len(truly_features)))
        dev_data = LazyLoadTensorDataset(truly_features, is_training=False)
        if args.local_rank == -1:
            dev_sampler = SequentialSampler(dev_data)
        else:
            dev_sampler = DistributedSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.val_batch_size)
        RawResult = collections.namedtuple("RawResult",
                                           ["unique_id", "logit"])

        model.eval()
        with torch.no_grad():
            cur_result = []
            for idx, batch in enumerate(tqdm(dev_dataloader, desc="predict interation: {}".format(args.dev_file))):
                # example index getter
                d_example_indices = batch[-1]
                # 多gpu训练的scatter
                if n_gpu == 1:
                    # 去除example index
                    batch = tuple(x.squeeze(0).to(device) for x in batch[:-1])
                else:
                    batch = batch[:-1]
                input_ids, input_mask, segment_ids, cls_mask, cls_weight = batch
                # 获取预测结果
                dev_logits = model(input_ids=input_ids,
                                   attention_mask=input_mask,
                                   token_type_ids=segment_ids,
                                   cls_mask=cls_mask,
                                   cls_weight=cls_weight)
                for i, example_index in enumerate(d_example_indices):
                    if args.model_name == 'RobertaForParagraphClassification':
                        dev_logit = dev_logits[i].detach().cpu().tolist()
                        dev_logit.reverse()
                    else:
                        dev_logit = dev_logits[i].detach().cpu().tolist()
                    dev_feature = truly_features[example_index.item()]
                    unique_id = dev_feature.unique_id
                    cur_result.append(RawResult(unique_id=unique_id,
                                             logit=dev_logit))

            tmp_best_paragraph, tmp_related_sentence, tmp_related_paragraph = write_predict_result(examples=truly_examples,
                                                                                                   features=truly_features,
                                                                                                   results=cur_result,
                                                                                                   has_sentence_result=has_sentence_result)
            all_results += cur_result
            best_paragraph.update(tmp_best_paragraph)
            related_sentence.update(tmp_related_sentence)
            related_paragraph.update(tmp_related_paragraph)
        del truly_examples, truly_features, dev_data
        gc.collect()
    # 获取新的文档
    logger.info("start saving data...")
    data = json.load(open(args.dev_file, 'r', encoding='utf-8'))
    # data = data[:100]
    for info in data:
        try:
            context = info['context']
            qas_id = info['_id']
            # (title, list(sent))
            get_best_paragraphs = context[best_paragraph[qas_id]]
            question = info['question']
            all_input_text = question + ''.join(get_best_paragraphs[1])
            # [10*predict, 10 * label]测试的时候无
            if has_sentence_result:
                get_sent_labels = related_sentence[qas_id]
                del_thread = sorted(get_sent_labels[0][:-1])
            del_idx = 0
            all_tokens = tokenizer.tokenize(all_input_text)
            cur_all_text = ''
            if len(all_tokens) <= 256:
                question += ''.join(get_best_paragraphs[1])
                cur_all_text = question
            else:
                while len(all_tokens) > 256:
                    mask = []
                    cur_all_text = question
                    for idx, paragraph in enumerate(get_best_paragraphs[1]):
                        if has_sentence_result:
                            if get_sent_labels[0][idx] > del_thread[del_idx]:
                                cur_all_text += paragraph
                                mask.append(1)
                            else:
                                mask.append(0)
                        else:
                            cur_all_text += paragraph
                            mask.append(1)
                    all_tokens = tokenizer.tokenize(cur_all_text)
                    if not has_sentence_result and len(all_tokens) > 256:
                        all_tokens = all_tokens[:256]
                    del_idx += 1
            all_tokens_len = len(tokenizer.tokenize(cur_all_text))
            total += all_tokens_len
            if all_tokens_len > 256:
                max_len += 1
            new_context[qas_id] = cur_all_text
        except Exception as e:
            import pdb; pdb.set_trace()
    logger.info("writing result to file...")
    if not os.path.exists(args.predict_result_path):
        logger.info("make new output dir:{}".format(args.predict_result_path))
        os.makedirs(args.predict_result_path)
    json.dump(best_paragraph, open("{}/{}".format(args.predict_result_path, args.best_paragraph_file),
                                   "w", encoding="utf-8"))
    json.dump(new_context, open("{}/{}".format(args.predict_result_path, args.new_context_file),
                                "w", encoding="utf-8"))
    json.dump(related_paragraph, open("{}/{}".format(args.predict_result_path, args.related_paragraph_file),
                                      "w", encoding='utf-8'))
    logger.info("write result done!")


if __name__ == '__main__':
    parser = get_config()
    args = parser.parse_args()
    run_predict(args=args)




