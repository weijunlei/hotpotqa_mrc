#!/usr/bin/evn python
# encoding: utf-8
'''
@author: xiaofenglei
@contact: weijunlei01@163.com
@file: train_qa_base.py
@time: 2020/9/7 17:56
@desc:
'''
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# """Run BERT on SQuAD."""
from __future__ import absolute_import, division, print_function
import argparse
import collections
import json
import logging
import math
import os
import random
import sys
from io import open
import re
import torch
import numpy as np
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Sampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import bisect

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from collections import Counter
import string
import gc

from origin_read_examples import read_examples, read_dev_examples
from origin_convert_example2features import convert_examples_to_features, convert_dev_examples_to_features
from origin_reader_helper import write_predictions, evaluate
from lazy_dataloader import LazyLoadTensorDataset
from config import get_config

sys.path.append("../pretrain_model")
from changed_model import BertForQuestionAnsweringCoAttention, \
    BertForQuestionAnsweringThreeCoAttention, \
    BertForQuestionAnsweringThreeSameCoAttention, \
    BertForQuestionAnsweringForward, \
    BertForQuestionAnsweringForwardBest
# from modeling_bert import *
from optimization import BertAdam, warmup_linear
from tokenization import (BasicTokenizer, BertTokenizer, whitespace_tokenize)

logger = None


def logger_config(log_path, log_prefix='lwj'):
    """
    日志配置
    :param log_path: 输出的日志路径
    :param log_prefix: 记录中的日志前缀
    :return:
    """
    logger = logging.getLogger(log_prefix)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def get_dev_data(args, tokenizer, logger=None):
    """ 获取验证集数据 """
    dev_examples = read_dev_examples(
        input_file=args.dev_file, filter_file=args.dev_filter_file, tokenizer=tokenizer, is_training=True)
    logger.info('dev examples: {}'.format(len(dev_examples)))
    dev_feature_file = args.dev_file.split('.')[0] + '_{0}_{1}_{2}_{3}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride),
        args.feature_suffix)
    if os.path.exists(dev_feature_file):
        with open(dev_feature_file, "rb") as reader:
            dev_features = pickle.load(reader)
    else:
        dev_features = convert_dev_examples_to_features(
            examples=dev_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            is_training=True)
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info("  Saving dev features into cached file %s", dev_feature_file)
            with open(dev_feature_file, "wb") as writer:
                pickle.dump(dev_features, writer)
    logger.info('dev feature_num: {}'.format(len(dev_features)))
    # d_all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
    # d_all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
    # d_all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
    # d_all_sent_mask = torch.tensor([f.sent_mask for f in dev_features], dtype=torch.long)
    # d_all_example_index = torch.arange(d_all_input_ids.size(0), dtype=torch.long)
    # d_all_content_len = torch.tensor([f.content_len for f in dev_features], dtype=torch.long)
    # dev_data = TensorDataset(d_all_input_ids, d_all_input_mask, d_all_segment_ids,
    #                          d_all_sent_mask, d_all_content_len, d_all_example_index)

    dev_data = LazyLoadTensorDataset(dev_features, is_training=False)
    if args.local_rank == -1:
        dev_sampler = RandomSampler(dev_data)
    else:
        dev_sampler = DistributedSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.val_batch_size)
    return dev_examples, dev_dataloader, dev_features


def get_train_data(args, tokenizer, logger=None):
    """ 获取训练数据 """
    cached_train_features_file = '{}/train_feature_file_{}_{}_{}_{}'.format(args.feature_cache_path,
                                                                            args.bert_model.split('/')[-1],
                                                                            str(args.max_seq_length),
                                                                            str(args.doc_stride),
                                                                            args.feature_suffix)
    if not os.path.exists(args.feature_cache_path):
        os.makedirs(args.feature_cache_path)
    tmp_cache_file = cached_train_features_file + '_' + str(0)
    if not os.path.exists(tmp_cache_file):
        train_examples = read_examples(
            input_file=args.train_file,
            filter_file=args.train_filter_file,
            tokenizer=tokenizer,
            is_training=True)
        # 当数据配置不变时可以设置为定值
        example_num = len(train_examples) # 89899
        random.shuffle(train_examples)
    else:
        example_num = 89541
    logger.info("train example num: {}".format(example_num))
    max_train_num = 200000
    start_idxs = list(range(0, example_num, max_train_num))
    end_idxs = [x + max_train_num for x in start_idxs]
    end_idxs[-1] = example_num
    total_feature_num = 0

    for i in range(len(start_idxs)):
        new_cache_file = cached_train_features_file + '_' + str(i)
        if os.path.exists(new_cache_file):
            with open(new_cache_file, "rb") as f:
                train_features = pickle.load(f)
        else:
            train_examples_ = train_examples[start_idxs[i]: end_idxs[i]]
            train_features = convert_examples_to_features(
                examples=train_examples_,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                is_training=True)
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("Saving train features into cached file {}".format(cached_train_features_file))
                with open(new_cache_file, "wb") as writer:
                    pickle.dump(train_features, writer)
        total_feature_num += len(train_features)
    logger.info('train feature_num: {}'.format(total_feature_num))
    return total_feature_num, start_idxs, cached_train_features_file


def dev_evaluate(model, dev_dataloader, n_gpu, device, dev_features, tokenizer, dev_examples):
    """ 模型验证 """
    model.eval()
    all_results = []
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logit", "end_logit", "sent_logit"])

    with torch.no_grad():
        for d_step, d_batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            d_example_indices = d_batch[-1].squeeze()
            if n_gpu == 1:
                d_batch = tuple(
                    t.squeeze(0).to(device) for t in d_batch[:-1])  # multi-gpu does scattering it-self
            else:
                d_batch = d_batch[:-1]
            d_all_input_ids, d_all_input_mask, d_all_segment_ids, \
            d_all_cls_mask, d_all_content_len = d_batch
            dev_start_logits, dev_end_logits, dev_sent_logits = model(d_all_input_ids, d_all_input_mask,
                                                                      d_all_segment_ids,
                                                                      sent_mask=d_all_cls_mask)
            for idx, example_index in enumerate(d_example_indices):
                dev_start_logit = dev_start_logits[idx].detach().cpu().tolist()
                dev_end_logit = dev_end_logits[idx].detach().cpu().tolist()
                dev_sent_logit = dev_sent_logits[idx].detach().cpu().tolist()
                dev_feature = dev_features[example_index.item()]
                unique_id = dev_feature.unique_id
                all_results.append(
                    RawResult(unique_id=unique_id, start_logit=dev_start_logit, end_logit=dev_end_logit,
                              sent_logit=dev_sent_logit))

    _, preds, sp_pred = write_predictions(tokenizer, dev_examples, dev_features, all_results)
    ans_f1, ans_em, sp_f1, sp_em, joint_f1, joint_em = evaluate(dev_examples, preds, sp_pred)
    model.train()
    return ans_f1, ans_em, sp_f1, sp_em, joint_f1, joint_em


def run_train():
    parser = get_config()
    args = parser.parse_args()
    # 配置日志文件
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    log_path = os.path.join(args.log_path, 'log_{}_{}_{}_{}_{}_{}.log'.format(args.log_prefix,
                                                                          args.bert_model.split('/')[-1],
                                                                          args.output_dir.split('/')[-1],
                                                                          args.train_batch_size,
                                                                          args.max_seq_length,
                                                                          args.doc_stride))
    logger = logger_config(log_path=log_path, log_prefix='')
    logger.info('-' * 15 + '所有配置' + '-' * 15)
    logger.info("所有参数配置如下：")
    for arg in vars(args):
        logger.info('{}: {}'.format(arg, getattr(args, arg)))
    logger.info('-' * 30)

    # 分布式训练设置
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

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    # 配置随机数
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    # 梯度积累设置
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    if not os.path.exists(args.train_file):
        raise ValueError("train file not exists! please set train file!")
    if not args.overwrite_result and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None

    # 自定义好的模型
    model_dict = {
        'BertForQuestionAnsweringCoAttention': BertForQuestionAnsweringCoAttention,
        'BertForQuestionAnsweringThreeCoAttention': BertForQuestionAnsweringThreeCoAttention,
        'BertForQuestionAnsweringThreeSameCoAttention': BertForQuestionAnsweringThreeSameCoAttention,
        'BertForQuestionAnsweringForward': BertForQuestionAnsweringForward,
        'BertForQuestionAnsweringForwardBest': BertForQuestionAnsweringForwardBest
    }
    model = model_dict[args.model_name].from_pretrained(args.bert_model)
    # 半精度和并行化使用设置
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

    # 参数配置
    param_optimizer = list(model.named_parameters())
    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    global_step = 0

    # 获取训练集数据
    total_feature_num, start_idxs, cached_train_features_file = get_train_data(args, tokenizer=tokenizer, logger=logger)
    model.train()
    num_train_optimization_steps = int(
        total_feature_num / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    max_f1 = 0
    print_loss = 0
    for epoch_idx in trange(int(args.num_train_epochs), desc="Epoch"):
        for ind in trange(len(start_idxs), desc='Data'):
            new_cache_file = cached_train_features_file + '_' + str(ind)
            logger.info("loading file: {}".format(new_cache_file))
            with open(new_cache_file, "rb") as reader:
                train_features = pickle.load(reader)
            logger.info("load file: {} done!".format(new_cache_file))
            # all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            # all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            # all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            # all_sent_mask = torch.tensor([f.sent_mask for f in train_features], dtype=torch.long)
            # all_sent_weight = torch.tensor([f.sent_weight for f in train_features], dtype=torch.float)
            # all_content_len = torch.tensor([f.content_len for f in train_features], dtype=torch.long)
            # all_start_position = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
            # all_end_position = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
            # all_sent_lbs = torch.tensor([f.sent_lbs for f in train_features], dtype=torch.long)
            # train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sent_mask,
            #                            all_sent_weight, all_content_len, all_start_position,
            #                            all_end_position, all_sent_lbs)
            train_data = LazyLoadTensorDataset(features=train_features, is_training=True)
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_data)
            else:
                train_sampler = DistributedSampler(train_data)

            train_dataloader = DataLoader(train_data,
                                          sampler=train_sampler,
                                          batch_size=args.train_batch_size,
                                          num_workers=4)
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if n_gpu == 1:
                    batch = tuple(t.squeeze(0).to(device) for t in batch)  # multi-gpu does scattering it-self
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, sent_mask, content_len, start_positions, end_positions, sent_lbs, sent_weight = batch
                loss, _, _, _ = model(input_ids,
                                      input_mask,
                                      segment_ids,
                                      start_positions=start_positions,
                                      end_positions=end_positions,
                                      sent_mask=sent_mask,
                                      sent_lbs=sent_lbs,
                                      sent_weight=sent_weight)
                if n_gpu > 1:
                    loss = loss.sum()  # mean() to average on multi-gpu.
                logger.debug("step = %d, train_loss=%f", global_step, loss)
                print_loss += loss
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if (global_step + 1) % 100 == 0 and (step + 1) % args.gradient_accumulation_steps == 0:
                    logger.info(
                        "epoch:{:3d},data:{:3d},global_step:{:8d},loss:{:8.3f}".format(epoch_idx, ind, global_step,
                                                                                       print_loss))
                    print_loss = 0
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used and handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                # 保存以及验证模型结果
                if (global_step + 1) % args.save_model_step == 0 and (step + 1) % args.gradient_accumulation_steps == 0:
                    # 获取验证集数据
                    dev_examples, dev_dataloader, dev_features = get_dev_data(args, tokenizer=tokenizer, logger=logger)
                    ans_f1, ans_em, sp_f1, sp_em, joint_f1, joint_em = dev_evaluate(model,
                                                                                    dev_dataloader,
                                                                                    n_gpu,
                                                                                    device,
                                                                                    dev_features,
                                                                                    tokenizer,
                                                                                    dev_examples)
                    logger.info("epoch:{} data: {} step: {}".format(epoch_idx, ind, global_step))
                    logger.info("ans_f1:{} ans_em:{} sp_f1:{} sp_em: {} joint_f1: {} joint_em:{}".format(
                        ans_f1, ans_em, sp_f1, sp_em, joint_f1, joint_em
                    ))
                    del dev_examples, dev_dataloader, dev_features
                    gc.collect()
                    logger.info("max_f1: {}".format(max_f1))
                    if joint_f1 > max_f1:
                        logger.info("get better model in step: {} with joint f1: {}".format(global_step, joint_f1))
                        max_f1 = joint_f1
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(args.output_dir, 'pytorch_model_{}.bin'.format(global_step))
                        torch.save(model_to_save.state_dict(), output_model_file)
                        output_config_file = os.path.join(args.output_dir, 'config.json')
                        with open(output_config_file, 'w') as f:
                            f.write(model_to_save.config.to_json_string())
                        logger.info('saving step: {} model'.format(global_step))
            # 内存清除
            del train_features, input_ids, input_mask, segment_ids
            del start_positions, end_positions, sent_lbs, sent_mask
            del sent_weight, train_data, train_dataloader
            gc.collect()


if __name__ == "__main__":
    run_train()
