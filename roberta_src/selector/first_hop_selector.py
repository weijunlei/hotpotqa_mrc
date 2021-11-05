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
"""Run BERT on SQuAD."""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import os
import random
import sys
import torch
from io import open
import numpy as np
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,Sampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import logging

from transformers import RobertaTokenizer
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import gc

sys.path.append("../pretrain_model")
from changed_model_roberta import RobertaForParagraphClassification, RobertaForRelatedSentence
from optimization import BertAdam, warmup_linear

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """
    def __init__(self,
                 qas_id,
                 question_tokens,
                 context_tokens,
                 sents_label=None,
                 para_related=None):
        self.qas_id = qas_id
        self.question_tokens=question_tokens
        self.context_tokens=context_tokens
        self.sents_label=sents_label
        self.para_related=para_related

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question: %s" % (
            self.question)
        s += ", sents: [%s]" % (" ".join(self.sents))
        if self.sents_label:
            s += ", sents_label: %d" % (self.sents_label)
        if self.para_related:
            s += ", para_related: %r" % (self.para_related)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_mask,
                 cls_label=None,
                 cls_weight=None,
                 is_related=None,
                 roll_back=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_mask=cls_mask
        self.cls_label=cls_label
        self.cls_weight=cls_weight
        self.is_related = is_related
        self.roll_back=roll_back


def read_examples(input_file, is_training):


    data = json.load(open(input_file, 'r'))
    examples = []
    pos=neg=0
    for d in data:
        context=d['context']
        question=d['question']
        sup=d['supporting_facts']
        for indc,con in enumerate(context):
            sents=con
            labels=[]
            related=False
            for inds,s in enumerate(sents[1]):
                if [sents[0],inds] in sup:
                    labels.append(1)
                    related=True
                else:
                    labels.append(0)
            if not related and random.random()>0.25:#为保证正负样本比例1:1
                continue
            if related:
                pos+=1
            else:
                neg+=1
            example = SquadExample(
                qas_id=d['_id']+'_'+str(indc),
                question_tokens=question,
                context_tokens=con,
                sents_label=labels,
                para_related=related
            )
            examples.append(example)
    print(pos,neg)
    return examples

def read_dev_examples(input_file, is_training):


    data = json.load(open(input_file, 'r'))
    examples = []
    for d in data:
        context=d['context']
        question=d['question']
        sup=d['supporting_facts']
        for indc,con in enumerate(context):
            sents=con
            labels=[]
            related=False
            for inds,s in enumerate(sents[1]):
                if [sents[0],inds] in sup:
                    labels.append(1)
                    related=True
                else:
                    labels.append(0)
            example = SquadExample(
                qas_id=d['_id']+'_'+str(indc),
                question_tokens=question,
                context_tokens=con,
                sents_label=labels,
                para_related=related
            )
            examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 sent_overlap, is_training):
    """Loads a data file into a list of `InputBatch`s."""
    features=[]
    related_sents = unrelated_sents = 0
    for (example_index, example) in tqdm(enumerate(examples)):
        query_tokens = tokenizer.tokenize(example.question_tokens)
        segment1_len=2+len(query_tokens)
        max_len=max_seq_length-len(query_tokens)-4
        length=0
        unique_id=0
        tokens=['<s>']+query_tokens+['</s>']+['</s>']
        cls_mask=[1]+[0]*(len(tokens)-1)
        cls_label=[1 if example.para_related else 0]+[0]*(len(tokens)-1)
        cls_weight = [1] + [0] * (len(tokens) - 1)
        i=0
        prev1=None

        while i<len(example.sents_label):
            sent=example.context_tokens[1][i]
            label=example.sents_label[i]
            sent_tokens=tokenizer.tokenize(sent)
            if label:
                related_sents+=1
            else:
                unrelated_sents+=1
            if len(sent_tokens)+1>max_len:
                sent_tokens=sent_tokens[:max_len-1]
            roll_back = 0
            if length+len(sent_tokens)+1>max_len:
                tokens+=['</s>']
                cls_mask += [1]
                cls_label += [0]
                cls_weight += [0]
                valid_len=len(tokens)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)+[1]*(max_seq_length-valid_len)
                segment_ids=[0]*segment1_len+[1]*(valid_len-segment1_len)+[0]*(max_seq_length-valid_len)
                input_mask=[1]*valid_len+[0]*(max_seq_length-valid_len)
                cls_mask+=[0]*(max_seq_length-valid_len)
                cls_label += [0] * (max_seq_length - valid_len)
                cls_weight += [0] * (max_seq_length - valid_len)
                if prev2 is not None:
                    if prev2+prev1+len(sent_tokens)+1<=max_len:
                        roll_back=2
                    elif prev1+len(sent_tokens)+1<=max_len:
                        roll_back=1
                elif prev1 is not None and prev1+len(sent_tokens)+1<=max_len:
                        roll_back=1
                i-=roll_back
                real_related=int(bool(sum(cls_label)-cls_label[0]))
                if real_related!=cls_label[0]:
                    cls_label[0]=real_related
                assert len(cls_mask) == max_seq_length
                assert len(cls_label) == max_seq_length
                assert len(cls_weight) == max_seq_length
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                features.append(
                    InputFeatures(
                        unique_id=example.qas_id + '_' + str(unique_id),
                        example_index=example_index,
                        doc_span_index=unique_id,
                        tokens=tokens,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        cls_mask=cls_mask,
                        cls_label=cls_label,
                        cls_weight=cls_weight,
                        is_related=real_related,
                        roll_back=roll_back))
                length = 0
                unique_id += 1
                tokens = ['<s>'] + query_tokens + ['</s>'] + ['</s>']
                cls_mask = [1] + [0] * (len(tokens) - 1)
                cls_label = [1 if example.para_related else 0] + [0] * (len(tokens) - 1)
                cls_weight = [1] + [0] * (len(tokens) - 1)
            else:
                tokens+=['<unk>']+sent_tokens
                cls_mask+=[1]+[0]*(len(sent_tokens)+0)
                cls_label+=[label]+[0]*(len(sent_tokens)+0)
                cls_weight += [1 if label else 0.2] + [0] * (len(sent_tokens) + 0)
                length+=len(sent_tokens)+1
                i+=1
            prev2 = prev1
            prev1 = len(sent_tokens) + 1
        tokens += ['</s>']
        cls_mask+=[1]
        cls_label+=[0]
        cls_weight+=[0]
        valid_len = len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens) + [1] * (max_seq_length - valid_len)
        segment_ids = [0] * segment1_len + [1] * (valid_len - segment1_len) + [0] * (max_seq_length - valid_len)
        input_mask = [1] * valid_len + [0] * (max_seq_length - valid_len)
        cls_mask += [0] * (max_seq_length - valid_len)
        cls_label += [0] * (max_seq_length - valid_len)
        cls_weight += [0] * (max_seq_length - valid_len)
        real_related = int(bool(sum(cls_label) - cls_label[0]))
        if real_related != cls_label[0]:
            cls_label[0] = real_related
        assert len(cls_mask) == max_seq_length
        assert len(cls_label) == max_seq_length
        assert len(cls_weight) == max_seq_length
        assert len(input_ids)==max_seq_length
        assert len(input_mask)==max_seq_length
        assert len(segment_ids) == max_seq_length
        features.append(
            InputFeatures(
                unique_id=example.qas_id + '_' + str(unique_id),
                example_index=example_index,
                doc_span_index=unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                cls_mask=cls_mask,
                cls_label=cls_label,
                cls_weight=cls_weight,
                is_related=real_related,
                roll_back=0))
    print(len(features))
    print(related_sents,unrelated_sents)
    return features

def write_predictions_(all_examples, all_features, all_results):
    """Write final predictions to the json file and log-odds of null if needed."""
    # logger.info("Writing predictions to: %s" % (output_prediction_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result[0]] = result
    para_results={}
    sent_results={}
    labels={}
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        id = '_'.join(features[0].unique_id.split('_')[:-1])
        if len(features)>1:
            roll_back=None
            para_result=0
            sent_result=[]
            lbs=[]
            overlap=0
            mask1=0
            for (feature_index, feature) in enumerate(features):
                rs=unique_id_to_result[feature.unique_id]
                if rs[1][0]>para_result:
                    para_result=rs[1][0]
                sr=[]
                lb = []
                mask1+=sum(feature.cls_mask[1:])
                for a,b,c in zip(feature.cls_mask[1:],rs[1][1:],feature.cls_label[1:]):
                    if a!=0:
                        lb.append(c)
                        sr.append(b)
                if roll_back is None:
                    roll_back=0
                    lbs.append(feature.cls_label[0])
                    lbs+=lb
                elif roll_back==1:
                    sent_result[-1]=max(sent_result[-1],sr[0])
                    sr=sr[1:]
                    if lbs[0]==0 and feature.cls_label[0]==1:
                        lbs[0]=1
                    lbs+=lb[1:]
                elif roll_back==2:
                    sent_result[-2] = max(sent_result[-2], sr[0])
                    sent_result[-1] = max(sent_result[-1], sr[1])
                    sr=sr[2:]
                    if lbs[0]==0 and feature.cls_label[0]==1:
                        lbs[0]=1
                    lbs+=lb[2:]
                else:
                    lbs+=lb
                sent_result+=sr
                overlap+=roll_back
                roll_back=feature.roll_back
            para_results[id]=para_result
            sent_results[id]=sent_result
            labels[id]=lbs
            assert len(sent_result)+overlap==mask1
            assert len(lbs) + overlap == mask1+1
        else:
            para_results[id]=unique_id_to_result[features[0].unique_id][1][0]
            sent_result=[]
            lbs=[]
            for ind,(a,b,c) in enumerate(zip(unique_id_to_result[features[0].unique_id][1],features[0].cls_mask,features[0].cls_label)):
                if ind==0:
                    lbs.append(c)
                    continue
                if b!=0:
                    lbs.append(c)
                    sent_result.append(a)
            sent_results[id]=sent_result
            labels[id]=lbs
            assert len(sent_result)==(sum(features[0].cls_mask)-1)
            assert len(lbs) == sum(features[0].cls_mask)

    q_para={}
    for k,v in para_results.items():
        try:
            q_para[k.split('_')[0]][0][int(k.split('_')[1])]=v
        except:
            q_para[k.split('_')[0]]=[[0]*10,[0]*10]
            q_para[k.split('_')[0]][0][int(k.split('_')[1])] = v
    for k,v in labels.items():
        q_para[k.split('_')[0]][1][int(k.split('_')[1])]=v[0]
    recall=count=precision=em=rec=acc=0
    for k,v in q_para.items():
        count+=1
        th=0.5
        p11=p10=p01=p00=0
        # print(th,v[0])
        maxlogit=-100
        maxrs=False
        vmax=max(v[0])
        vmin=min(v[0])
        # maxpara=-1
        for indab,(a,b) in enumerate(zip(v[0],v[1])):
            if a>maxlogit:
                maxlogit=a
                if b==1:
                    maxrs=True
                # maxpara=indab
            a=(a-vmin)/(vmax-vmin)
            a = 1 if a > th else 0
            if a==1 and b==1:
                p11+=1
            elif a==1 and b==0:
                p10+=1
            elif a==0 and b==1:
                p01+=1
            elif a==0 and b==0:
                p00+=1
        if p11+p01!=0:
            recall+=p11/(p11+p01)
        else:
            print('error')
        if p11+p10==0:
            print('error')
        else:
            precision+=p11/(p11+p10)
        if p11==2 and p10==0:
            em+=1
        if p01==0:
            rec+=1
        if maxrs:
            acc+=1
    return acc/count,precision/count,em/count,rec/count


def logging(s, config,print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open( config.output_log, 'a+') as f_log:
            f_log.write(s + '\n')


def run_train():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default='roberta_large', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default='../../data/checkpoints/selector/20211105_origin_roberta', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--model_name", type=str, default='RobertaForRelatedSentence',
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--train_file", default='../../data/hotpot_data/hotpot_train_labeled_data_v3.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--dev_file", default='../../data/hotpot_data/hotpot_dev_labeled_data_v3.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--sent_overlap", default=2, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--train_batch_size", default=64, type=int, help="Total batch size for training.")
    parser.add_argument("--val_batch_size", default=128, type=int, help="Total batch size for validation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--output_log", type=str, default='selector_round1_large_1e-5.txt', )
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true', default=True,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--save_model_step',
                        type=int, default=500,
                        help="The proportion of the validation set")
    args = parser.parse_args()

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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.train_file:
        raise ValueError(
            "If `do_train` is True, then `train_file` must be specified.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # preprocess_data

    tokenizer = RobertaTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None

    # Prepare model
    models= {'RobertaForRelatedSentence': RobertaForRelatedSentence,
            'RobertaForParagraphClassification': RobertaForParagraphClassification}
    model = models[args.model_name].from_pretrained(args.bert_model)

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
        input_file=args.dev_file, is_training=True)
    print('dev example_num:', len(dev_examples))
    dev_feature_file = args.dev_file.split('.')[0] + '_r1_{0}_{1}_{2}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.sent_overlap))
    try:
        with open(dev_feature_file, "rb") as reader:
            dev_features = pickle.load(reader)
    except:
        dev_features = convert_examples_to_features(
            examples=dev_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            sent_overlap=args.sent_overlap,
            is_training=True)
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info("  Saving dev features into cached file %s", dev_feature_file)
            with open(dev_feature_file, "wb") as writer:
                pickle.dump(dev_features, writer)
    print('dev feature_num:', len(dev_features))
    d_all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
    d_all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
    d_all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
    d_all_cls_mask = torch.tensor([f.cls_mask for f in dev_features], dtype=torch.long)
    d_all_cls_label = torch.tensor([f.cls_label for f in dev_features], dtype=torch.long)
    d_all_cls_weight = torch.tensor([f.cls_weight for f in dev_features], dtype=torch.float)
    d_all_example_index = torch.arange(d_all_input_ids.size(0), dtype=torch.long)
    dev_data = TensorDataset(d_all_input_ids, d_all_input_mask, d_all_segment_ids,
                             d_all_cls_mask, d_all_cls_label, d_all_cls_weight, d_all_example_index)
    if args.local_rank == -1:
        dev_sampler = RandomSampler(dev_data)
    else:
        dev_sampler = DistributedSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.val_batch_size)

    cached_train_features_file = args.train_file.split('.')[0] + '_r1_{0}_{1}_{2}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.sent_overlap))
    train_features = None
    model.train()
    train_examples = read_examples(
        input_file=args.train_file,is_training=True)
    example_num=len(train_examples)
    print('train example_num:', example_num)
    start = list(range(0, example_num, 215000))
    end = []
    for i in start:
        end.append(i + 215000)
    end[-1] = example_num
    print(len(start))
    total_feature_num=0
    random.shuffle(train_examples)
    for i in range(len(start)):
        train_examples_=train_examples[start[i]:end[i]]
        try:
            with open(cached_train_features_file + '_' + str(i), "rb") as reader:
                train_features = pickle.load(reader)
        except:
            train_features = convert_examples_to_features(
                examples=train_examples_,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                sent_overlap=args.sent_overlap,
                is_training=True)
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving train features into cached file %s", cached_train_features_file)
                with open(cached_train_features_file + '_' + str(i), "wb") as writer:
                    pickle.dump(train_features, writer)
        total_feature_num+=len(train_features)
    print('train feature_num:', total_feature_num)  # 168148

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
    # RawResult = collections.namedtuple("RawResult",
    #                                    ["unique_id", "start_pos", "end_pos", "start_logit", "end_logit"])
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "logit"])
    mmax=0
    printloss=0
    ls=len(start)
    for ee in trange(int(args.num_train_epochs), desc="Epoch"):
        for ind in trange(ls,desc='Data'):
            with open(cached_train_features_file+'_'+str(ind), "rb") as reader:
                train_features = pickle.load(reader)
            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_cls_mask = torch.tensor([f.cls_mask for f in train_features], dtype=torch.long)
            all_cls_label = torch.tensor([f.cls_label for f in train_features], dtype=torch.long)
            all_cls_weight = torch.tensor([f.cls_weight for f in train_features], dtype=torch.float)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                       all_cls_mask,all_cls_label,all_cls_weight)
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_data)
            else:
                train_sampler = DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler,batch_size=args.train_batch_size)
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if n_gpu == 1:
                    batch = tuple(t.squeeze(0).to(device) for t in batch)  # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, cls_mask,cls_label,cls_weight = batch

                loss,_= model(input_ids, input_mask,segment_ids, cls_mask=cls_mask,cls_label=cls_label,cls_weight=cls_weight)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                logger.info("  step = %d, train_loss=%f", global_step, loss)
                printloss += loss
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if (global_step+1)%100==0 and (step + 1) % args.gradient_accumulation_steps == 0:
                    logging("epoch:{:3d},data:{:3d},global_step:{:8d},loss:{:8.3f}".format(ee,ind,global_step,printloss),args)
                    printloss=0
                if (global_step+1)%args.save_model_step==0 and (step + 1) % args.gradient_accumulation_steps == 0:
                    model.eval()
                    all_results = []
                    total_loss = 0

                    with torch.no_grad():
                        for d_step, d_batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
                            d_example_indices=d_batch[-1]
                            if n_gpu == 1:
                                d_batch = tuple(t.squeeze(0).to(device) for t in d_batch[:-1])  # multi-gpu does scattering it-self
                            d_all_input_ids, d_all_input_mask, d_all_segment_ids,d_all_cls_mask, d_all_cls_label, d_all_cls_weight=d_batch[:-1]
                            dev_loss, dev_logits= model(d_all_input_ids,d_all_input_mask,d_all_segment_ids,
                             cls_mask=d_all_cls_mask, cls_label=d_all_cls_label,cls_weight=d_all_cls_weight)
                            dev_loss=torch.sum(dev_loss)
                            dev_logits=torch.sigmoid(dev_logits)
                            total_loss += dev_loss
                            # print(dev_logits.shape)
                            for i, example_index in enumerate(d_example_indices):
                                # start_position = start_positions[i].detach().cpu().tolist()
                                # end_position = end_positions[i].detach().cpu().tolist()
                                dev_logit = dev_logits[i].detach().cpu().tolist()
                                dev_feature = dev_features[example_index.item()]
                                unique_id = dev_feature.unique_id
                                all_results.append(RawResult(unique_id=unique_id,
                                                             logit=dev_logit))

                    acc,prec,em,rec = write_predictions_(dev_examples, dev_features, all_results)
                    # pickle.dump(all_results, open('all_results.pkl', 'wb'))
                    logging("epoch={:3d},data={:3d},step={:8d},max_acc={:4.8f},precision={:4.8f},em={:4.8f},recall={:4.8f},max_max={:4.8f},loss={:8.3f}".format(ee, ind, global_step,acc,prec,em,rec,mmax,total_loss),args)

                    if acc >mmax:
                        mmax=acc
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Only save the model it-self
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
            del train_features,all_input_ids,all_input_mask,all_segment_ids,all_cls_label,all_cls_mask,all_cls_weight,train_data,train_dataloader
            gc.collect()



if __name__ == "__main__":
    run_train()
