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
from transformers.tokenization_bert import (BertTokenizer)
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
sys.path.append("../pretrain_model")
from changed_model import BertForRelatedSentence, BertForParagraphClassification
from modeling_bert import *
from optimization import BertAdam, warmup_linear
from tokenization import BertTokenizer

# 日志设置
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def write_second_predict_result(examples, features, results, has_sentence_result=True):
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
                for get_idx, (label_result, cls_mask) in enumerate(zip(labels_result, cls_masks)):
                    if get_idx == 0:
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
                    for get_idx, (label_result, cls_mask) in enumerate(zip(label_results, cls_masks)):
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
            context_dict[context_id] = [[0]*10,[0]*10]
        context_dict[context_id][0][paragraph_id] = v
        # 在预测时只做标记位，为1标记可以被选中，为0不能被选中
        context_dict[context_id][1][paragraph_id] = 1
    # 将context最大结果导出
    for k, v in context_dict.items():
        thread = 0.01
        max_v = max(v[0])
        min_v = min(v[0])
        max_logit = -1000
        max_result = False
        get_related_paras = []
        max_para = -1
        for para_idx, para_pro in enumerate(v[0]):
            # 后面的筛选条件保证两个段落不会相同
            if para_pro > max_logit and v[1][para_idx] != 0:
                max_logit = para_pro
                max_para = para_idx
            if max_v - min_v == 0:
                para_pro = 0
            else:
                para_pro = (para_pro - min_v) / (max_v - min_v)
            if para_pro > thread and v[1][para_idx] != 0:
                get_related_paras.append(para_idx)
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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=args.do_lower_case)

    # Prepare model
    models_dict = {"BertForRelatedSentence": BertForRelatedSentence,
                   "BertForParagraphClassification": BertForParagraphClassification}
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
    dev_examples = read_second_hotpotqa_examples(input_file=args.dev_file,
                                                 best_paragraph_file="{}/{}".format(args.first_predict_result_path,
                                                                                    args.best_paragraph_file),
                                                 related_paragraph_file="{}/{}".format(args.first_predict_result_path,
                                                                                       args.related_paragraph_file),
                                                 new_context_file="{}/{}".format(args.first_predict_result_path,
                                                                                 args.new_context_file),

                                                 is_training='test')
    example_num = len(dev_examples)
    max_train_data_size = 100000
    start_idxs = list(range(0, example_num, max_train_data_size))
    end_idxs = [x + max_train_data_size for x in start_idxs]
    end_idxs[-1] = example_num
    all_results = []
    best_paragraph = {}
    related_sentence = {}
    related_paragraph = {}
    total = 0
    max_len = 0

    data = json.load(open(args.dev_file, 'r', encoding='utf-8'))
    # data = data[:100]
    first_hop_dict = {}
    first_best_paragraph_file = "{}/{}".format(args.first_predict_result_path, args.best_paragraph_file)
    first_best_paragraph = json.load(open(first_best_paragraph_file, 'r', encoding='utf-8'))
    for info in data:
        title = info['context'][first_best_paragraph[info['_id']]][0]
        first_hop_dict[info['_id']] = 0
        for supporting_fact in info['supporting_facts']:
            if title == supporting_fact[0]:
                first_hop_dict[info['_id']] = 1
                break

    for start_idx in range(len(start_idxs)):
        logger.info("start idx: {} all idx length: {}".format(start_idx, len(start_idxs)))
        truly_examples = dev_examples[start_idxs[start_idx]: end_idxs[start_idx]]
        truly_features = convert_examples_to_second_features(
            examples=truly_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            is_training='test'
        )
        logger.info("all truly gotten features: {}".format(len(truly_features)))
        d_all_input_ids = torch.tensor([f.input_ids for f in truly_features], dtype=torch.long)
        d_all_input_mask = torch.tensor([f.input_mask for f in truly_features], dtype=torch.long)
        d_all_segment_ids = torch.tensor([f.segment_ids for f in truly_features], dtype=torch.long)
        d_all_cls_mask = torch.tensor([f.cls_mask for f in truly_features], dtype=torch.long)
        d_all_cls_weight = torch.tensor([f.cls_weight for f in truly_features], dtype=torch.float)
        d_all_example_index = torch.arange(d_all_input_ids.size(0), dtype=torch.long)
        dev_data = TensorDataset(d_all_input_ids, d_all_input_mask, d_all_segment_ids,
                                 d_all_cls_mask, d_all_cls_weight, d_all_example_index)
        if args.local_rank == -1:
            dev_sampler = SequentialSampler(dev_data)
        else:
            dev_sampler = DistributedSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.val_batch_size)
        RawResult = collections.namedtuple("RawResult",
                                           ["unique_id", "logit"])

        model.eval()
        has_sentence_result = True
        if args.model_name == 'BertForParagraphClassification':
            has_sentence_result = False
        with torch.no_grad():
            cur_result = []
            for batch_idx, batch in enumerate(tqdm(dev_dataloader, desc="predict interation: {}".format(args.dev_file))):
                # example index getter
                d_example_indices = batch[-1]
                # 多gpu训练的scatter
                if n_gpu == 1:
                    # 去除example index
                    batch = tuple(x.squeeze(0).to(device) for x in batch[:-1])
                else:
                    batch = batch[:-1]
                d_all_input_ids, d_all_input_mask, d_all_segment_ids, d_all_cls_mask, d_all_cls_weight = batch
                # 获取预测结果
                dev_logits = model(d_all_input_ids, d_all_input_mask, d_all_segment_ids,
                                   cls_mask=d_all_cls_mask, cls_weight=d_all_cls_weight)
                for i, example_index in enumerate(d_example_indices):
                    # start_position = start_positions[i].detach().cpu().tolist()
                    # end_position = end_positions[i].detach().cpu().tolist()
                    if args.model_name == 'BertForParagraphClassification':
                        dev_logit = dev_logits[i].detach().cpu().tolist()
                        dev_logit.reverse()
                    else:
                        dev_logit = dev_logits[i].detach().cpu().tolist()
                    dev_feature = truly_features[example_index.item()]
                    unique_id = dev_feature.unique_id
                    cur_result.append(RawResult(unique_id=unique_id,
                                             logit=dev_logit))
            tmp_best_paragraph, tmp_related_sentence, tmp_related_paragraph = write_second_predict_result(
                examples=truly_examples,
                features=truly_features,
                results=cur_result,
                has_sentence_result=has_sentence_result
            )
            all_results += cur_result
            best_paragraph.update(tmp_best_paragraph)
            related_sentence.update(tmp_related_sentence)
            related_paragraph.update(tmp_related_paragraph)
        del truly_examples, truly_features, dev_data
        gc.collect()
    logger.info("writing result to file...")
    if not os.path.exists(args.second_predict_result_path):
        logger.info("make new output dir:{}".format(args.second_predict_result_path))
        os.makedirs(args.second_predict_result_path)
    final_related_paragraph_file = "{}/{}".format(args.second_predict_result_path, args.final_related_result)
    final_second_related_paragraph_file = ""
    final_related_paragraph_dict = {}
    for k, v in best_paragraph.items():
        final_related_paragraph_dict[k] = [first_best_paragraph[k], best_paragraph[k]]
    logger.info(len(final_related_paragraph_dict))
    json.dump(final_related_paragraph_dict, open(final_related_paragraph_file, 'w', encoding='utf-8'))
    logger.info("write result done!")

    # write third context
    new_context = {}
    data = json.load(open(args.dev_file, 'r', encoding='utf-8'))
    # data = data[:100]
    over_half_num = 0
    for info in data:
        context = info['context']
        qas_id = info['_id']
        # (title, list(sent))
        get_best_paragraphs = context[final_related_paragraph_dict[qas_id][1]]
        get_second_paragraphs = context[final_related_paragraph_dict[qas_id][0]]
        question = info['question']
        # 1 指示句子s
        all_input_text = question + ''.join(get_best_paragraphs[1])
        all_input_text += ''.join(get_second_paragraphs[1])
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
            over_half_num += 1
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
                if has_sentence_result:
                    del_idx += 1
        all_tokens_len = len(tokenizer.tokenize(cur_all_text))
        total += all_tokens_len
        if all_tokens_len > 256:
            max_len += 1
        new_context[qas_id] = cur_all_text
    logger.info("over half length number: {}".format(over_half_num))
    json.dump(new_context, open("{}/{}".format(args.second_predict_result_path, args.new_context_file),
                                "w", encoding="utf-8"))
    json.dump(related_paragraph, open("{}/{}".format(args.second_predict_result_path, args.related_paragraph_file),
                                      "w", encoding='utf-8'))
    logger.info("write result done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--checkpoint_path", default='../checkpoints/selector/second_hop_selector', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--model_name", type=str, default='BertForRelated',
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--dev_file", default='../data/hotpot_data/hotpot_train_labeled_data_v2.json', type=str,
                        help="dev file")
    # parser.add_argument("--pred_output", type=str, default='round1_base/train_preds.json',
    #                     help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--first_predict_result_path", default="../data/selector/first_hop_result/", type=str,
                        help="The output directory of all result")
    parser.add_argument("--second_predict_result_path", default="../data/selector/second_hop_result/", type=str,
                        help="The output directory of all result")
    parser.add_argument("--final_related_result", default="train_related.json", type=str,
                        help="The output directory of all result")
    parser.add_argument("--best_paragraph_file", default='train_best_paragraph.json', type=str,
                        help="best_paragraph_file")
    parser.add_argument("--related_paragraph_file", default='train_related_paragraph.json', type=str,
                        help="related_paragraph_file")
    parser.add_argument("--new_context_file", default='train_new_context.json', type=str,
                        help="new_context_file")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--sent_overlap", default=2, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--val_batch_size", default=128, type=int, help="Total batch size for validation.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--output_log", type=str, default='military_output_log.txt', )
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
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
    args = parser.parse_args()
    run_predict(args=args)