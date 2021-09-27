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
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,Sampler,
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
from config import get_config
sys.path.append("../pretrain_model")
from modeling_bert import *
from optimization import BertAdam, warmup_linear
from tokenization import (BasicTokenizer,BertTokenizer,whitespace_tokenize)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def logging(s, config, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(config.output_log, 'a+') as f_log:
            f_log.write(s + '\n')


def run_train():
    args = get_config()
    # 配置随机数
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.train_file:
        raise ValueError(
            "If `do_train` is True, then `train_file` must be specified.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None

    # Prepare model
    model_dict = {
            'BertForQuestionAnsweringCoAttention': BertForQuestionAnsweringCoAttention,
            'BertForQuestionAnsweringThreeCoAttention': BertForQuestionAnsweringThreeCoAttention,
            'BertForQuestionAnsweringThreeSameCoAttention': BertForQuestionAnsweringThreeSameCoAttention,
            'BertForQuestionAnsweringForward': BertForQuestionAnsweringForward,
    }
    model = model_dict[args.model_name].from_pretrained('bert-base-uncased')

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

    # Prepare optimizer
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

    dev_examples = read_dev_examples(
        input_file=args.dev_file,filter_file=args.dev_filter_file, tokenizer=tokenizer,is_training=True)
    print('dev example_num:',len(dev_examples))#3243
    dev_feature_file = args.dev_file.split('.')[0] + '_{0}_{1}_{2}_{3}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride),args.feature_suffix)
    # dev_feature_file_ = args.dev_file.split('.')[0] + '_{0}_{1}_{2}_{3}'.format(
    #     list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride), 'shi')
    try:
        with open(dev_feature_file, "rb") as reader:
            dev_features = pickle.load(reader)
        # print(dev_feature_file_)
        # with open(dev_feature_file_, "rb") as reader:
        #     dev_features_ = pickle.load(reader)
        # print('fine')
        # assert dev_features==dev_features_
    except:
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
    print('dev feature_num:', len(dev_features))
    d_all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
    d_all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
    d_all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
    d_all_sent_mask = torch.tensor([f.sent_mask for f in dev_features], dtype=torch.long)
    d_all_example_index = torch.arange(d_all_input_ids.size(0), dtype=torch.long)
    d_all_content_len = torch.tensor([f.content_len for f in dev_features], dtype=torch.long)
    dev_data = TensorDataset(d_all_input_ids, d_all_input_mask, d_all_segment_ids,
                             d_all_sent_mask, d_all_content_len,d_all_example_index)
    if args.local_rank == -1:
        dev_sampler = RandomSampler(dev_data)
    else:
        dev_sampler = DistributedSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.val_batch_size)

    cached_train_features_file = args.train_file.split('.')[0] + '_{0}_{1}_{2}_{3}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride), args.feature_suffix)
    # cached_train_features_file_ = args.train_file.split('.')[0] + '_{0}_{1}_{2}_{3}'.format(
    #     list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride), 'shi')
    train_features = None
    model.train()
    train_examples = read_examples(
        input_file=args.train_file,
        filter_file=args.train_filter_file,
        tokenizer=tokenizer,
        is_training=True)
    example_num = len(train_examples)
    print('train example_num:', example_num)
    max_train_num = 215000
    start = list(range(0, example_num, max_train_num))
    end = []
    for i in start:
        end.append(i + max_train_num)
    end[-1] = example_num
    print(len(start))
    total_feature_num = 0
    random.shuffle(train_examples)
    for i in range(len(start)):
        try:
            with open(cached_train_features_file + '_' + str(i), "rb") as reader:
                train_features = pickle.load(reader)
            # with open(cached_train_features_file_ + '_' + str(i), "rb") as reader:
            #     train_features_ = pickle.load(reader)
        except:
            train_examples_ = train_examples[start[i]:end[i]]
            train_features = convert_examples_to_features(
                examples=train_examples_,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                is_training=True)
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving train features into cached file %s", cached_train_features_file)
                with open(cached_train_features_file + '_' + str(i), "wb") as writer:
                    pickle.dump(train_features, writer)
        total_feature_num+=len(train_features)
    print('train feature_num:', total_feature_num)
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
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logit","end_logit","sent_logit"])
    max_f1 = 0
    printloss = 0
    ls=len(start)
    for ee in trange(int(args.num_train_epochs), desc="Epoch"):
        for ind in trange(ls,desc='Data'):
            with open(cached_train_features_file+'_'+str(ind), "rb") as reader:
                train_features = pickle.load(reader)
            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_start_position = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
            all_end_position = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
            all_sent_mask = torch.tensor([f.sent_mask for f in train_features], dtype=torch.long)
            all_sent_lbs = torch.tensor([f.sent_lbs for f in train_features], dtype=torch.long)
            all_sent_weight = torch.tensor([f.sent_weight for f in train_features], dtype=torch.float)
            # all_mask = torch.tensor([f.mask for f in train_features], dtype=torch.long)
            all_content_len=torch.tensor([f.content_len for f in train_features], dtype=torch.long)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,all_start_position,all_end_position,
                                       all_sent_mask,all_sent_lbs,all_sent_weight,all_content_len)
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_data)
            else:
                train_sampler = DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if n_gpu == 1:
                    batch = tuple(t.squeeze(0).to(device) for t in batch)  # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids,start_position,end_position, sent_mask,sent_lbs,sent_weight,content_len = batch
                loss, _, _, _ = model(input_ids,
                                      input_mask,
                                      segment_ids,
                                      start_positions=start_position,
                                      end_positions=end_position,
                                      sent_mask=sent_mask,
                                      sent_lbs=sent_lbs,
                                      sent_weight=sent_weight)
                if n_gpu > 1:
                    loss = loss.sum()  # mean() to average on multi-gpu.
                logger.info("step = %d, train_loss=%f", global_step, loss)
                printloss += loss
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if (global_step+1) % 100 == 0 and (step + 1) % args.gradient_accumulation_steps == 0:
                    logging("epoch:{:3d},data:{:3d},global_step:{:8d},loss:{:8.3f}".format(ee, ind, global_step, printloss), args)
                    printloss = 0
                if (global_step+1)%args.save_model_step == 0 and (step + 1) % args.gradient_accumulation_steps == 0:
                    model.eval()
                    all_results = []
                    total_loss = 0

                    with torch.no_grad():
                        for d_step, d_batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
                            d_example_indices=d_batch[-1].squeeze()
                            if n_gpu == 1:
                                d_batch = tuple(t.squeeze(0).to(device) for t in d_batch[:-1])  # multi-gpu does scattering it-self
                            else:
                                d_batch=d_batch[:-1]
                            d_all_input_ids, d_all_input_mask, d_all_segment_ids, \
                            d_all_cls_mask,d_all_content_len=d_batch
                            dev_start_logits, dev_end_logits,dev_sent_logits= model(d_all_input_ids, d_all_input_mask, d_all_segment_ids, sent_mask=d_all_cls_mask)
                            # total_loss += dev_loss
                            for i, example_index in enumerate(d_example_indices):
                                # start_position = start_positions[i].detach().cpu().tolist()
                                # end_position = end_positions[i].detach().cpu().tolist()
                                dev_start_logit = dev_start_logits[i].detach().cpu().tolist()
                                dev_end_logit = dev_end_logits[i].detach().cpu().tolist()
                                dev_sent_logit = dev_sent_logits[i].detach().cpu().tolist()
                                dev_feature = dev_features[example_index.item()]
                                unique_id = dev_feature.unique_id
                                all_results.append(RawResult(unique_id=unique_id,start_logit=dev_start_logit,end_logit=dev_end_logit,sent_logit=dev_sent_logit))

                    _, preds, sp_pred = write_predictions(tokenizer,dev_examples, dev_features, all_results)
                    ans_f1, ans_em, sp_f1, sp_em, joint_f1,joint_em=evaluate(dev_examples,preds,sp_pred)
                    # pickle.dump(all_results, open('all_results.pkl', 'wb'))
                    logging("epoch={:3d}, data={:3d},step = {:6d},ans_f1={:4.8f},ans_em={:4.8f},sp_f1={:4.8f},sp_em={:4.8f},joint_f1={:4.8f},joint_em={:4.8f},max_f1={:4.8f},total_loss={:8.3f}" \
                            .format(ee, ind, global_step,ans_f1,ans_em,sp_f1,sp_em,joint_f1,joint_em,max_f1,total_loss) ,args)
                    if joint_f1 > max_f1:
                        max_f1=joint_f1
                        model_to_save = model.module if hasattr(model,'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(args.output_dir, 'pytorch_model.bin')
                        torch.save(model_to_save.state_dict(), output_model_file)
                        output_config_file = os.path.join(args.output_dir, 'config.json')
                        with open(output_config_file, 'w') as f:
                            f.write(model_to_save.config.to_json_string())
                        logger.info('saving model')
                    model.train()
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
            # 内存清除
            del train_features, all_input_ids, all_input_mask, all_segment_ids
            del all_start_position, all_end_position, all_sent_lbs, all_sent_mask
            del all_sent_weight, train_data, train_dataloader
            gc.collect()


if __name__ == "__main__":
    run_train()