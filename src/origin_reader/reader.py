#!/usr/bin/evn python
# encoding: utf-8
'''
@author: xiaofenglei
@contact: weijunlei01@163.com
@file: train_qa_base.py.py
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
import numpy as np
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,Sampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
sys.path.append("../pretrain_model")
from modeling_bert import *
from optimization import BertAdam, warmup_linear
from tokenization import (BasicTokenizer,BertTokenizer,whitespace_tokenize)
import bisect
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from collections import Counter
import string
import gc
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 orig_tokens,
                 doc_tokens,
                 sub_to_orig_index,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 sent_cls=None,
                 sent_lbs=None,
                 full_sents_mask=None,
                 full_sents_lbs=None,
                 mask_matrix=None,
                 subwords_to_matrix=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.orig_tokens=orig_tokens
        self.sub_to_orig_index=sub_to_orig_index
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.sent_cls=sent_cls
        self.sent_lbs=sent_lbs
        self.full_sents_mask = full_sents_mask
        self.full_sents_lbs = full_sents_lbs
        self.mask_matrix=mask_matrix
        self.subwords_to_matrix=subwords_to_matrix

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 sent_mask=None,
                 sent_lbs=None,
                 sent_weight=None,
                 mask=None,
                 content_len=None
                 ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.sent_mask=sent_mask
        self.sent_lbs=sent_lbs
        self.sent_weight=sent_weight
        self.mask=mask
        self.content_len=content_len


def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它 # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

def find_nearest(a, target, test_func=lambda x: True):
    idx = bisect.bisect_left(a, target)
    if (0 <= idx < len(a)) and a[idx] == target:
        return target, 0
    elif idx == 0:
        return a[0], abs(a[0] - target)
    elif idx == len(a):
        return a[-1], abs(a[-1] - target)
    else:
        d1 = abs(a[idx] - target) if test_func(a[idx]) else 1e200
        d2 = abs(a[idx-1] - target) if test_func(a[idx-1]) else 1e200
        if d1 > d2:
            return a[idx-1], d2
        else:
            return a[idx], d1

def fix_span(para, offsets, span):
    span = span.strip()
    parastr = " ".join(para)

    assert span in parastr, '{}\t{}'.format(span, parastr)
    # print([y for x in offsets for y in x])
    begins=[]
    ends=[]
    for o in offsets:
        begins.append(o[0])
        ends.append(o[1])
    # begins, ends = map(list, zip([y for x in offsets for y in x]))#在列表前加*号，会将列表拆分成一个一个的独立元素

    best_dist = 1e200
    best_indices = None

    if span == parastr:
        return parastr, (0, len(parastr)), 0

    for m in re.finditer(re.escape(span), parastr):
        begin_offset, end_offset = m.span()
        fixed_begin, d1 = find_nearest(begins, begin_offset, lambda x: x < begin_offset)
        fixed_end, d2 = find_nearest(ends, end_offset, lambda x: x > end_offset)

        if d1 + d2 < best_dist:
            best_dist = d1 + d2
            best_indices = (fixed_begin, fixed_end)
            if best_dist == 0:
                break

    assert best_indices is not None
    return parastr[best_indices[0]:best_indices[1]], best_indices, best_dist

def read_examples(input_file, filter_file,tokenizer,is_training):
    filter = json.load(open(filter_file, 'r'))
    examples = []
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c=='\xa0':
            return True
        return False

    fail_count=0
    lines=json.load(open(input_file,'r',encoding='utf-8'))
    fail=0
    for d in lines:
        context = ""
        question = d['question']
        answer=d['answer']
        answer_label=d['labels'][0]
        sup = d['supporting_facts']
        id=d['_id']
        length=len(context)
        sent_cls=[]
        sent_lbs=[]
        start_position=None
        end_position=None
        if answer.lower()=='yes':
            start_position=-1
            end_position=-1
        if answer.lower()=='no':
            start_position=-2
            end_position=-2
        full_sents_mask=[]
        full_sents_lbs=[]
        char_to_matrix=[]
        sid=1
        for ind_con,con in enumerate(d['context']):#为了去掉句首的空白
            if ind_con in filter[id]:
                full_sents_mask+=[1 for con1 in con[1] if con1.strip()!='']
            else:
                full_sents_mask+=[0]*len(con[1])
            for indc1,c1 in enumerate(con[1]):
                if ind_con in filter[id] and c1.strip()=='':
                    continue
                if [con[0],indc1] in sup:
                    full_sents_lbs.append(1)
                else:
                    full_sents_lbs.append(0)
            if con[0]==answer_label[0] and ind_con not in filter[id]:
                fail+=1
                break
            if ind_con in filter[id]:
                offset=0
                added=0
                for inds,sent in enumerate(con[1]):
                    if sent.strip()=='':
                        continue
                    if context=='':
                        white1=0
                        while is_whitespace(sent[0]):
                            white1+=1
                            sent=sent[1:]
                        offset+=white1
                    elif not sent.startswith(' '):
                        context+=' '
                        length+=1
                        char_to_matrix += [sid]
                    # context+='<unk>'+sent
                    context +=sent
                    char_to_matrix+=[sid]*len(sent)
                    sid+=1
                    sent_cls.append(length+0)
                    if con[0]==answer_label[0]:
                        if answer_label[1]>=offset and answer_label[1]<offset+len(sent):
                            start_position=length+answer_label[1]-offset#+5
                        # if answer_label[2]>=offset-white1 and answer_label[2]<=offset:
                        #     end_position=length
                        if answer_label[2]>offset and answer_label[2]<=offset+len(sent):
                            end_position=length+answer_label[2]-offset#+5
                    length += len(sent)# + 5
                    offset+=len(sent)
                    if [con[0],inds] in sup:
                        sent_lbs.append(1)
                    else:
                        sent_lbs.append(0)
        if start_position is None:
            continue

        doc_tokens = []#为了消除多个whitespace连在一起
        char_to_newchar = []
        prev_is_whitespace = True
        char_count=-2
        for c in context:
            if is_whitespace(c):
                prev_is_whitespace = True
                char_to_newchar.append(char_count+1)
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                    char_count+=2
                else:
                    doc_tokens[-1] += c
                    char_count+=1
                prev_is_whitespace = False
                char_to_newchar.append(char_count)
        if prev_is_whitespace:
            char_to_newchar.append(char_count+1)

        context=' '.join(doc_tokens)
        sent_cls_n = []
        # print(id)
        newchar_to_matrix=[0]*len(context)
        for ind_ctm,ctm in enumerate(char_to_matrix):
            if char_to_newchar[ind_ctm]>=len(context):
                continue
            newchar_to_matrix[char_to_newchar[ind_ctm]]=ctm
        for sc in sent_cls:
            sent_cls_n.append(char_to_newchar[sc])
        start_position_n = char_to_newchar[start_position]
        if end_position == len(char_to_newchar):
            end_position_n = char_to_newchar[end_position - 1] + 1
        else:
            end_position_n = char_to_newchar[end_position]

        # an=context[start_position_n:end_position_n]
        # if an!=answer and answer!='yes' and answer!='no':
        #     print('error')
        char_to_word_offset = []
        subwords_to_matrix=[]
        doc_subwords=[]
        sub_to_orig_index=[]
        # cur_subword=0
        # cur_subword_offset=0
        conlen = 0
        for indt, dtoken in enumerate(doc_tokens):
            sub_tokens = tokenizer.tokenize(dtoken)
            sum_toklen = 0
            unkc = 0
            conlen += len(dtoken)
            for indst, subtoken in enumerate(sub_tokens):
                tok_len = len(subtoken)
                doc_subwords.append(subtoken)
                sub_to_orig_index.append(indt)
                if len(char_to_word_offset)<len(newchar_to_matrix):
                    subwords_to_matrix.append(newchar_to_matrix[len(char_to_word_offset)])
                else:
                    subwords_to_matrix.append(newchar_to_matrix[len(char_to_word_offset)-1])
                if subtoken.startswith('##'):
                    tok_len -= 2
                if subtoken == '[UNK]':
                    unkc += 1
                    if len(sub_tokens) == indst + 1:
                        tok_len = len(dtoken) - sum_toklen
                    elif sub_tokens[indst + 1] == '[UNK]':
                        tok_len = 1
                    else:
                        tok_len = dtoken.find(sub_tokens[indst + 1][0], sum_toklen) - sum_toklen
                prelen=len(char_to_word_offset)
                for rr in range(tok_len):
                    if len(char_to_word_offset) < conlen:
                        char_to_word_offset.append(len(doc_subwords) - 1)
                # while len(char_to_word_offset) > conlen:
                #     char_to_word_offset = char_to_word_offset[:-1]
                sum_toklen += len(char_to_word_offset)-prelen
            if indt != len(doc_tokens) - 1:
                char_to_word_offset.append(len(doc_subwords))
                conlen+=1
            while len(char_to_word_offset) < conlen:
                char_to_word_offset.append(len(doc_subwords) - 1)
            while len(char_to_word_offset) > conlen:
                char_to_word_offset = char_to_word_offset[:-1]
            assert conlen == len(char_to_word_offset)
        assert len(char_to_word_offset) == len(context)

        sent_cls_w=[]
        for sc in sent_cls_n:
            sent_cls_w.append(char_to_word_offset[sc])
        start_position_w=char_to_word_offset[start_position_n]
        if end_position_n==len(char_to_word_offset):
            end_position_w=char_to_word_offset[end_position_n-1]+1
        else:
            end_position_w=char_to_word_offset[end_position_n]
        sent_cls_extend=[]
        for ind_scw,scw in enumerate(sent_cls_w):
            doc_subwords.insert(scw+ind_scw,'[UNK]')
            sub_to_orig_index.insert(scw+ind_scw,sub_to_orig_index[scw+ind_scw])
            subwords_to_matrix.insert(scw+ind_scw,subwords_to_matrix[scw+ind_scw])
            sent_cls_extend.append(scw+ind_scw)
            if start_position_w>=scw+ind_scw:
                start_position_w+=1
            if end_position_w>scw+ind_scw:
                end_position_w+=1

        # tok_text = " ".join(doc_subwords[start_position_w:end_position_w])
        # tok_text=tok_text.replace('##','')
        # tok_text=tok_text.replace(' ##','')
        # tok_text=tok_text.strip()
        # orig_text=" ".join(doc_tokens[sub_to_orig_index[start_position_w]:sub_to_orig_index[end_position_w-1]+1])
        # actual_text=get_final_text(tok_text, orig_text, True, False)
        # tok_text_f = ''.join(tok_text.split())
        # actual_text_f=''.join(actual_text.split()).lower()
        # start_offset=actual_text_f.find(tok_text_f)
        # end_offset=len(actual_text_f)-start_offset-len(tok_text_f)
        # if start_offset>=0:
        #     if end_offset!=0:
        #         actual_text=actual_text[start_offset:-end_offset]
        #     else:
        #         actual_text=actual_text[start_offset:]
        # # actual_text=tokenizer.convert_tokens_to_string(doc_subwords[start_position_w:end_position_w]).strip()
        # cleaned_answer_text = " ".join(whitespace_tokenize(answer))
        #
        # if actual_text!=cleaned_answer_text and answer!='yes' and answer!='no':
        #
        #     print(actual_text)
        #     print(cleaned_answer_text)
        #     print()

        if answer.lower()=='yes':
            start_position_w=-1
            end_position_w=-1
        if answer.lower()=='no':
            start_position_w=-2
            end_position_w=-2
        example = SquadExample(
            qas_id=id,
            question_text=question,
            orig_tokens=doc_tokens,
            sub_to_orig_index=sub_to_orig_index,
            doc_tokens=doc_subwords,
            orig_answer_text=answer,
            start_position=start_position_w,
            end_position=end_position_w,
            sent_cls=sent_cls_extend,
            sent_lbs=sent_lbs,
            full_sents_mask=full_sents_mask,
            full_sents_lbs=full_sents_lbs,
            subwords_to_matrix=subwords_to_matrix
        )
        examples.append(example)
    print('fail:',fail_count)
    # logging(input_file+' fail count '+str(fail_count))
    return examples

def read_dev_examples(input_file, filter_file,tokenizer,is_training):
    filter = json.load(open(filter_file, 'r'))
    examples = []
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c=='\xa0':
            return True
        return False

    fail_count=0
    lines=json.load(open(input_file,'r',encoding='utf-8'))
    fail=0
    for d in lines:
        context = ""
        question = d['question']
        answer=d['answer']
        # answer_label=d['labels'][0]
        sup = d['supporting_facts']
        id=d['_id']
        length=len(context)
        sent_cls=[]
        full_sents_mask=[]
        full_sents_lbs=[]
        char_to_matrix = []
        sid = 1
        for ind_con,con in enumerate(d['context']):#为了去掉句首的空白
            if ind_con in filter[id]:
                full_sents_mask+=[1 for con1 in con[1] if con1.strip()!='']
            else:
                full_sents_mask+=[0]*len(con[1])
            for indc1,c1 in enumerate(con[1]):
                if ind_con in filter[id] and c1.strip()=='':
                    continue
                if [con[0],indc1] in sup:
                    full_sents_lbs.append(1)
                else:
                    full_sents_lbs.append(0)
            # if con[0]==answer_label[0] and ind_con not in filter[id]:
            #     fail+=1
            #     break
            if ind_con in filter[id]:
                offset=0
                for inds,sent in enumerate(con[1]):
                    if sent.strip()=='':
                        continue
                    if context=='':
                        white1=0
                        while is_whitespace(sent[0]):
                            white1+=1
                            sent=sent[1:]
                        offset+=white1
                    elif not sent.startswith(' '):
                        context+=' '
                        length+=1
                        char_to_matrix+=[sid]
                    # context+='<unk>'+sent
                    context +=sent
                    char_to_matrix += [sid] * len(sent)
                    sid += 1
                    sent_cls.append(length+0)
                    length += len(sent)# + 5
                    offset+=len(sent)
        # 'Ġ'
        # an0=context[start_position:end_position]
        # if start_position!=-1 and start_position!=-2:
        #     end_position=start_position+len(answer)

        doc_tokens = []#为了消除多个whitespace连在一起
        char_to_newchar = []
        prev_is_whitespace = True
        char_count=-2
        for c in context:
            if is_whitespace(c):
                prev_is_whitespace = True
                char_to_newchar.append(char_count+1)
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                    char_count+=2
                else:
                    doc_tokens[-1] += c
                    char_count+=1
                prev_is_whitespace = False
                char_to_newchar.append(char_count)
        if prev_is_whitespace:
            char_to_newchar.append(char_count+1)

        context=' '.join(doc_tokens)
        sent_cls_n = []
        newchar_to_matrix = [0] * len(context)
        for ind_ctm, ctm in enumerate(char_to_matrix):
            newchar_to_matrix[char_to_newchar[ind_ctm]] = ctm
        # print(id)
        for sc in sent_cls:
            sent_cls_n.append(char_to_newchar[sc])

        # an=context[start_position_n:end_position_n]
        # if an!=answer and answer!='yes' and answer!='no':
        #     print('error')
        char_to_word_offset = []
        subwords_to_matrix=[]
        doc_subwords=[]
        sub_to_orig_index=[]
        # cur_subword=0
        # cur_subword_offset=0
        conlen=0
        for indt, dtoken in enumerate(doc_tokens):
            sub_tokens = tokenizer.tokenize(dtoken)
            sum_toklen = 0
            unkc = 0
            conlen += len(dtoken)
            for indst, subtoken in enumerate(sub_tokens):
                tok_len = len(subtoken)
                doc_subwords.append(subtoken)
                sub_to_orig_index.append(indt)
                if len(char_to_word_offset)<len(newchar_to_matrix):
                    subwords_to_matrix.append(newchar_to_matrix[len(char_to_word_offset)])
                else:
                    subwords_to_matrix.append(newchar_to_matrix[len(char_to_word_offset)-1])
                if subtoken.startswith('##'):
                    tok_len -= 2
                if subtoken == '[UNK]':
                    unkc += 1
                    if len(sub_tokens) == indst + 1:
                        tok_len = len(dtoken) - sum_toklen
                    elif sub_tokens[indst + 1] == '[UNK]':
                        tok_len = 1
                    else:
                        tok_len = dtoken.find(sub_tokens[indst + 1][0], sum_toklen) - sum_toklen
                prelen=len(char_to_word_offset)
                for rr in range(tok_len):
                    if len(char_to_word_offset) < conlen:
                        char_to_word_offset.append(len(doc_subwords) - 1)
                # while len(char_to_word_offset) > conlen:
                #     char_to_word_offset = char_to_word_offset[:-1]
                sum_toklen += len(char_to_word_offset)-prelen
            if indt != len(doc_tokens) - 1:
                char_to_word_offset.append(len(doc_subwords))
                conlen+=1
            while len(char_to_word_offset) < conlen:
                char_to_word_offset.append(len(doc_subwords) - 1)
            while len(char_to_word_offset) > conlen:
                char_to_word_offset = char_to_word_offset[:-1]
            assert conlen == len(char_to_word_offset)
        assert len(char_to_word_offset) == len(context)
        assert len(char_to_word_offset) == len(context)
        # for indc,c in enumerate(context):
        #     char_to_word_offset.append(cur_subword)
        #     cur_subword_offset += len(c.encode('utf-8'))
        #     while cur_subword_offset>=len(doc_subwords[cur_subword]):
        #         cur_subword_offset-=len(doc_subwords[cur_subword])
        #         subwords_to_matrix.append(newchar_to_matrix[indc])
        #         cur_subword += 1
        #         if cur_subword>=len(doc_subwords):
        #             break
        sent_cls_w=[]
        for sc in sent_cls_n:
            sent_cls_w.append(char_to_word_offset[sc])
        sent_cls_extend=[]
        for ind_scw,scw in enumerate(sent_cls_w):
            doc_subwords.insert(scw+ind_scw,'[UNK]')
            sub_to_orig_index.insert(scw+ind_scw,sub_to_orig_index[scw+ind_scw])
            subwords_to_matrix.insert(scw+ind_scw,subwords_to_matrix[scw+ind_scw])
            sent_cls_extend.append(scw+ind_scw)
        if 0 in subwords_to_matrix:
            print('error')
        # actual_text = "".join(doc_subwords[start_position_w:end_position_w]).replace('Ġ',' ').strip()
        # cleaned_answer_text = " ".join(whitespace_tokenize(answer))
        # if actual_text!=cleaned_answer_text and answer!='yes':
        #     print(actual_text)
        #     print(cleaned_answer_text)
        #     print()

        example = SquadExample(
            qas_id=id,
            question_text=question,
            orig_tokens=doc_tokens,
            sub_to_orig_index=sub_to_orig_index,
            doc_tokens=doc_subwords,
            orig_answer_text=answer,
            sent_cls=sent_cls_extend,
            full_sents_mask=full_sents_mask,
            full_sents_lbs=full_sents_lbs,
            subwords_to_matrix=subwords_to_matrix
        )
        examples.append(example)
    print('fail:',fail_count)
    # logging(input_file+' fail count '+str(fail_count))
    return examples



def convert_examples_to_features(examples, tokenizer,graph, max_seq_length,
                                 doc_stride,  is_training):
    """Loads a data file into a list of `InputBatch`s."""
    # full_graph=json.load(open(graph,'r'))
    unique_id = 1000000000
    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)
        all_doc_tokens = example.doc_tokens
        # graph=full_graph[example.qas_id]
        # The -5 accounts for '<s>','yes','no', </s> and </s>
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 5

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = ["[CLS]","yes","no"]
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = [0,0,0]
            matrix=[0,0,0]

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = split_token_index
                is_max_context = _check_is_max_context(doc_spans, doc_span_index,split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                matrix.append(example.subwords_to_matrix[split_token_index])
                segment_ids.append(0)
            content_len=len(tokens)
            tokens.append("[SEP]")
            segment_ids.append(0)
            matrix.append(0)

            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
            matrix += [0] * len(query_tokens) + [-1]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                matrix.append(-1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(matrix)==max_seq_length

            # mask = []
            # last = -1
            # for mi in range(max_seq_length):
            #     if matrix[mi] == -1:
            #         continue
            #     if matrix[mi] == last:
            #         mask[-1][-1] += 1
            #         continue
            #     last = matrix[mi]
            #     cur_mask = []
            #     prev = False
            #     for mj in range(max_seq_length):
            #         if matrix[mi] == -1 or matrix[mj] == -1:
            #             prev = False
            #             continue
            #         if graph[matrix[mi]][matrix[mj]] == 1:
            #             if not prev:
            #                 cur_mask.append([mi, mj, 1])
            #             else:
            #                 cur_mask[-1][-1] += 1
            #             prev = True
            #         else:
            #             prev = False
            #     cur_mask.append(1)
            #     mask.append(cur_mask)
            new_mask = []
            # for ma in mask:
            #     for maa in ma:
            #         if isinstance(maa, list):
            #             new_mask.append(maa + [ma[-1]])
            # while len(new_mask) < max_seq_length:
            #     new_mask.append([0, 0, 0, 0])

            start_position_f=None
            end_position_f=None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length
                if example.start_position==-1 and example.end_position==-1:
                    start_position_f=1
                    end_position_f=1
                elif example.start_position==-2 and example.end_position==-2:
                    start_position_f=2
                    end_position_f=2
                else:
                    if example.start_position>=doc_start and example.end_position<=doc_end:
                        start_position_f=example.start_position-doc_start+3
                        end_position_f=example.end_position-doc_start+2
                    else:
                        start_position_f=0
                        end_position_f=0
                sent_mask=[0]*max_seq_length
                sent_lbs=[0]*max_seq_length
                sent_weight=[0]*max_seq_length
                for ind_cls,orig_cls in enumerate(example.sent_cls):
                    if orig_cls>=doc_start and orig_cls<doc_end:
                        sent_mask[orig_cls-doc_start+3]=1
                        if example.sent_lbs[ind_cls]==1:
                            sent_lbs[orig_cls-doc_start+3]=1
                            sent_weight[orig_cls-doc_start+3]=1
                        else:
                            sent_weight[orig_cls - doc_start + 3] = 0.5

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position_f,
                    end_position=end_position_f,
                    sent_mask=sent_mask,
                    sent_lbs=sent_lbs,
                    sent_weight=sent_weight,
                    mask=new_mask,
                    content_len=content_len
                    ))
            unique_id += 1
    return features

def convert_dev_examples_to_features(examples, tokenizer,graph, max_seq_length,
                                 doc_stride,  is_training):
    """Loads a data file into a list of `InputBatch`s."""
    # full_graph = json.load(open(graph, 'r'))
    unique_id = 1000000000
    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)
        all_doc_tokens = example.doc_tokens
        # graph=full_graph[example.qas_id]
        # The -5 accounts for '<s>','yes','no', </s> and </s>
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 5

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = ["[CLS]","yes","no"]
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = [0,0,0]
            matrix=[0,0,0]

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = split_token_index
                is_max_context = _check_is_max_context(doc_spans, doc_span_index,split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(0)
                matrix.append(example.subwords_to_matrix[split_token_index])
            content_len=len(tokens)
            tokens.append("[SEP]")
            segment_ids.append(0)
            matrix.append(-1)

            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
            matrix += [0] * len(query_tokens) + [-1]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                matrix.append(-1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(matrix)==max_seq_length
            sent_mask = [0] * max_seq_length
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length
            for ind_cls, orig_cls in enumerate(example.sent_cls):
                if orig_cls >= doc_start and orig_cls < doc_end:
                    sent_mask[orig_cls - doc_start + 3] = 1

            mask=[]
            # last=-1
            # for mi in range(max_seq_length):
            #     if matrix[mi]==-1:
            #         continue
            #     if matrix[mi]==last:
            #         mask[-1][-1]+=1
            #         continue
            #     last=matrix[mi]
            #     cur_mask=[]
            #     prev = False
            #     for mj in range(max_seq_length):
            #         if matrix[mi] == -1 or matrix[mj] == -1:
            #             prev=False
            #             continue
            #         if graph[matrix[mi]][matrix[mj]] == 1:
            #             if not prev:
            #                 cur_mask.append([mi, mj, 1])
            #             else:
            #                 cur_mask[-1][-1] += 1
            #             prev=True
            #         else:
            #             prev=False
            #     cur_mask.append(1)
            #     mask.append(cur_mask)
            new_mask=[]
            # for ma in mask:
            #     for maa in ma:
            #         if isinstance(maa,list):
            #             new_mask.append(maa+[ma[-1]])
            # while len(new_mask) < max_seq_length:
            #     new_mask.append([0,0,0,0])

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    sent_mask=sent_mask,
                    mask=new_mask,
                    content_len=content_len))
            unique_id += 1
    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def write_predictions_(tokenizer,all_examples, all_features, all_results, n_best_size=20,
                      max_answer_length=20, do_lower_case=True):
    """Write final predictions to the json file and log-odds of null if needed."""
    # logger.info("Writing predictions to: %s" % (output_prediction_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    sp_preds={}
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        sent_pred_logit=[0.0]*len(example.doc_tokens)
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            # offset=feature.input_ids.index(102)+1
            start_indexes = _get_best_indexes(result.start_logit, n_best_size)
            end_indexes = _get_best_indexes(result.end_logit, n_best_size)
            for ind_rsl,rsl in enumerate(result.sent_logit):
                if feature.sent_mask[ind_rsl]==1 and feature.token_is_max_context.get(ind_rsl,False):
                    sent_pred_logit[feature.token_to_orig_map[ind_rsl]]=rsl
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    # start_index+=offset
                    # end_index+=offset
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logit[start_index],
                            end_logit=result.end_logit[end_index]))
        # prelim_predictions = sorted(
        #     prelim_predictions,
        #     key=lambda x: (x.start_logit + x.end_logit),
        #     reverse=True)

        sent_pred_logit=[spl for ind_spl,spl in enumerate(sent_pred_logit) if ind_spl in example.sent_cls]
        sp_pred=[]
        pointer=0
        # print(len(sent_pred_logit))
        # print(sum(example.full_sents_mask))
        for fsm in example.full_sents_mask:
            if fsm==0:
                sp_pred.append(0.0)
            else:
                sp_pred.append(sent_pred_logit[pointer])
                pointer+=1
        sp_preds[example.qas_id]=sp_pred
        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["start", "end", "text", "start_logit",'end_logit'])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                tok_tokens=[tt for tt in tok_tokens if tt!='[UNK]']
                orig_doc_start = example.sub_to_orig_index[feature.token_to_orig_map[pred.start_index]]
                orig_doc_end = example.sub_to_orig_index[feature.token_to_orig_map[pred.end_index]]
                orig_tokens = example.orig_tokens[orig_doc_start:(orig_doc_end + 1)]
                orig_tokens=[ot for ot in orig_tokens if ot!='[UNK]']

                tok_text = " ".join(tok_tokens)
                tok_text = tok_text.replace("##", "")
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                # tok_text=tokenizer.convert_tokens_to_string(tok_tokens)

                orig_text = " ".join(orig_tokens)
                # orig_text = orig_text.replace("##", "").strip()
                # orig_text = orig_text.strip()
                # orig_text = " ".join(orig_text.split())
                # orig_text=tokenizer.convert_tokens_to_string(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, False)
                tok_text_f = ''.join(tok_text.split())
                final_text_f = ''.join(final_text.split()).lower()
                start_offset = final_text_f.find(tok_text_f)
                end_offset = len(final_text_f) - start_offset - len(tok_text_f)
                if start_offset >= 0:
                    if end_offset != 0:
                        final_text = final_text[start_offset:-end_offset]
                    else:
                        final_text = final_text[start_offset:]
                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    start=orig_doc_start,
                    end=orig_doc_end,
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        # if not nbest:
        #     nbest.append(
        #         _NbestPrediction(start=0, end=0, text="", logit=-1e6))
        # nbest.append(_NbestPrediction(start=0, end=0, text="", logit=result.start_logit[0]+result.end_logit[0]))
        nbest.append(_NbestPrediction(start=1, end=1, text="yes", start_logit=result.start_logit[1],end_logit= result.end_logit[1]))
        nbest.append(_NbestPrediction(start=2, end=2, text="no", start_logit=result.start_logit[2],end_logit=result.end_logit[2]))
        nbest = sorted(nbest,key=lambda x: (x.start_logit+x.end_logit),reverse=True)

        # assert len(nbest) >= 1
        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit+entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        all_predictions[example.qas_id] = nbest_json[0]

    return nbest_json,all_predictions,sp_preds

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall

def update_sp(prediction, gold):
    tp, fp, fn = 0, 0, 0
    for p,g in zip(prediction,gold):
        # if p==0.0:
        #     if g==1:
        #         fn+=1
        # else:
        #     if p[0]<p[1] and g==1:
        #         tp+=1
        #     if p[0]<p[1] and g==0:
        #         fp+=1
        #     if p[0]>p[1] and g==1:
        #         fn+=1
        if p>0.5 and g==1:
            tp+=1
        if p>0.5 and g==0:
            fp+=1
        if p<=0.5 and g==1:
            fn+=1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    return f1,em, prec, recall

def eval(prediction_file, gold_file):
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}
    for dp in gold:
        cur_id = dp['_id']
        can_eval_joint = True
        if cur_id not in prediction['answer']:
            print('missing answer {}'.format(cur_id))
            can_eval_joint = False
        else:
            em, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], dp['answer'])
        if cur_id not in prediction['sp']:
            print('missing sp fact {}'.format(cur_id))
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, prediction['sp'][cur_id], dp['supporting_facts'])

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em

            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N

    print(metrics)

def evaluate(eval_examples, answer_dict,sp_preds):

    ans_f1 = ans_em = sp_f1=sp_em=joint_f1=joint_em=total = 0
    for ee in eval_examples:
        pred=answer_dict[ee.qas_id]['text']
        ans=ee.orig_answer_text
        total+=1
        # print(pred)
        # print(ans)
        # print()
        a_f1,a_prec,a_recall=f1_score(pred,ans)
        ans_f1+=a_f1
        a_em=exact_match_score(pred,ans)
        ans_em+=a_em
        s_f1,s_em,s_prec,s_recall=update_sp(sp_preds[ee.qas_id],ee.full_sents_lbs)
        sp_f1+=s_f1
        sp_em+=s_em
        j_prec = a_prec * s_prec
        j_recall = a_recall * s_recall
        if j_prec + j_recall > 0:
            j_f1 = 2 * j_prec * j_recall / (j_prec + j_recall)
        else:
            j_f1 = 0.
        j_em = a_em * s_em
        joint_f1+=j_f1
        joint_em+=j_em
    return ans_f1/total,ans_em/total,sp_f1/total,sp_em/total,joint_f1/total,joint_em/total

def logging(s, config,print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open( config.output_log, 'a+') as f_log:
            f_log.write(s + '\n')


def run_train():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default='bert-base-cased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default='../../data/checkpoints/qa_base_20210923_origin_coattention', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--model_name", type=str, default='BertForQuestionAnsweringCoAttention',
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--train_file", default='../../data/hotpot_data/hotpot_train_labeled_data_v3.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--dev_file", default='../../data/hotpot_data/hotpot_dev_distractor_v1.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--train_filter_file", default='../../data/selector/second_hop_related_paragraph_result/train_related.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--dev_filter_file", default='../../data/selector/second_hop_related_paragraph_result/dev_related.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--train_graph_file", default='../../data/graph_base/train_graph.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--dev_graph_file", default='../../data/graph_base/dev_graph.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--feature_suffix", default='graph3oqsur', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=256, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--val_batch_size", default=128, type=int, help="Total batch size for validation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--output_log", type=str, default='gra36e-5_full_.txt', )
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
                        type=int, default=1000,
                        help="The proportion of the validation set")
    args = parser.parse_args()
    #output_dir,train_file,max_seq_length,doc_stride,max_query_length,validate_proportion, train_batch_size,val_train_size,learning_rate,warmup_proportion,save_model_step

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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None

    # Prepare model
    models={'BertForQuestionAnsweringGraph': BertForQuestionAnsweringGraph,
            'BertForQuestionAnsweringCoAttention': BertForQuestionAnsweringCoAttention,
            'BertForQuestionAnsweringThreeCoAttention': BertForQuestionAnsweringThreeCoAttention,
            'BertForQuestionAnsweringThreeSameCoAttention': BertForQuestionAnsweringThreeSameCoAttention}
    model = models[args.model_name].from_pretrained('bert-base-uncased')

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
            graph=args.dev_graph_file,
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
    # d_all_mask = torch.tensor([f.mask for f in dev_features], dtype=torch.long)
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
        list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride),args.feature_suffix)
    # cached_train_features_file_ = args.train_file.split('.')[0] + '_{0}_{1}_{2}_{3}'.format(
    #     list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride), 'shi')
    train_features = None
    model.train()
    train_examples = read_examples(
        input_file=args.train_file,filter_file=args.train_filter_file, tokenizer=tokenizer,is_training=True)
    example_num = len(train_examples)
    # example_num=86034#fake_train
    print('train example_num:', example_num)
    start = list(range(0, example_num, 215000))
    end = []
    for i in start:
        end.append(i + 215000)
    end[-1] = example_num
    print(len(start))
    total_feature_num = 90001
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
                graph=args.train_graph_file,
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
    # RawResult = collections.namedtuple("RawResult",
    #                                    ["unique_id", "start_pos", "end_pos", "start_logit", "end_logit"])
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
            train_dataloader = DataLoader(train_data, sampler=train_sampler,batch_size=args.train_batch_size)
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if n_gpu == 1:
                    batch = tuple(t.squeeze(0).to(device) for t in batch)  # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids,start_position,end_position, sent_mask,sent_lbs,sent_weight ,content_len = batch

                # if len(masks.shape)!=3:
                #     masks=masks.unsqueeze(0)
                # bsz = input_ids.shape[0]
                # real_masks=np.zeros((bsz,args.max_seq_length,args.max_seq_length))
                # stop=torch.zeros((1,4)).long()
                # for iter in range(bsz):
                #     for indm in range(args.max_seq_length):
                #         if torch.sum(stop.eq(masks[iter][indm])).item()==4:
                #             break
                #         real_masks[iter][masks[iter][indm][0].item():masks[iter][indm][0].item()+masks[iter][indm][-1].item(),masks[iter][indm][1].item():masks[iter][indm][1].item()+masks[iter][indm][-2].item()]=1
                # real_masks=torch.from_numpy(real_masks).cuda()

                loss, _, _, _ = model(input_ids, input_mask,segment_ids, start_positions=start_position,end_positions=end_position,
                              sent_mask=sent_mask,sent_lbs=sent_lbs,sent_weight=sent_weight,mask=None)
                if n_gpu > 1:
                    loss = loss.sum() # mean() to average on multi-gpu.
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
                            d_example_indices=d_batch[-1].squeeze()
                            if n_gpu == 1:
                                d_batch = tuple(t.squeeze(0).to(device) for t in d_batch[:-1])  # multi-gpu does scattering it-self
                            else:
                                d_batch=d_batch[:-1]
                            d_all_input_ids, d_all_input_mask, d_all_segment_ids, \
                            d_all_cls_mask,d_all_content_len=d_batch
                            d_real_masks = []
                            # d_masks = d_all_mask.detach().cpu().tolist()
                            d_content_len = d_all_content_len.detach().cpu().tolist()
                            # for indm,d_mask in enumerate(d_masks):
                            #     d_real_mask = np.zeros((args.max_seq_length, args.max_seq_length))
                            #     for ma in d_mask:
                            #         # print(ma)
                            #         if ma == [0, 0, 0, 0]:
                            #             break
                            #         d_real_mask[ma[0]: ma[0]+ ma[-1], ma[1]:ma[1] + ma[-2]] = 1
                            #     # d_real_mask[0:d_content_len[indm], 0:3] = 1 - d_real_mask[0:d_content_len[indm], 0:3]
                            #     d_real_masks.append(d_real_mask)
                            # d_real_masks = torch.tensor(d_real_masks, dtype=torch.long).cuda()
                            dev_start_logits,dev_end_logits,dev_sent_logits= model(d_all_input_ids,d_all_input_mask,d_all_segment_ids,sent_mask=d_all_cls_mask,mask=None)
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

                    _,preds,sp_pred = write_predictions_(tokenizer,dev_examples, dev_features, all_results)
                    ans_f1,ans_em,sp_f1,sp_em,joint_f1,joint_em=evaluate(dev_examples,preds,sp_pred)
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
            del train_features,all_input_ids,all_input_mask,all_segment_ids,all_start_position,all_end_position,all_sent_lbs,all_sent_mask,all_sent_weight,train_data,train_dataloader
            gc.collect()



if __name__ == "__main__":
    run_train()