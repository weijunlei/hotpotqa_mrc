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
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Sampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import bisect
import pickle
from collections import Counter
import string
import gc

from origin_reader_helper import HotpotQAReaderExample,\
    HotpotQAInputFeatures,\
    cut_sent,\
    fix_span,\
    find_nearest,\
    _improve_answer_span,\
    _check_is_max_context,\
    get_final_text,\
    _get_best_indexes,\
    _compute_softmax, \
    evaluate

from origin_read_examples import reader_read_examples
from origin_convert_example_to_features import convert_examples_to_features
sys.path.append("../pretrain_model")
from modeling_bert import *
from optimization import BertAdam, warmup_linear
from tokenization import BertTokenizer

# 日志设置
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def logging(s, config, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open( config.output_log, 'a+') as f_log:
            f_log.write(s + '\n')


def write_predictions(all_examples,
                      all_features,
                      all_results,
                      n_best_size=20,
                      max_answer_length=20,
                      do_lower_case=True):
    """ 根据预测导出结果 """
    example_index2features = collections.defaultdict(list)
    for feature in all_features:
        example_index2features[feature.example_index].append(feature)
    unique_id2result = {}
    for result in all_results:
        unique_id2result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])
    all_predictions = collections.OrderedDict()
    supporting_fact_preds = {}
    for example_idx, example in enumerate(all_examples):
        features = example_index2features[example_idx]
        prelim_predictions = []
        sentence_pred_logit = [0.0] * len(example.doc_tokens)
        for feature_idx, feature in enumerate(features):
            result = unique_id2result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logit, n_best_size=n_best_size)
            end_indexes = _get_best_indexes(result.end_logit, n_best_size=n_best_size)
            for sentence_logit_idx, sentence_logit in enumerate(result.sent_logit):
                if feature.sentence_mask[sentence_logit_idx] == 1 and feature.token_is_max_context.get(sentence_logit_idx, False):
                    sentence_pred_logit[feature.token2origin_map[sentence_logit_idx]] = sentence_logit
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token2origin_map:
                        continue
                    if end_index not in feature.token2origin_map:
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
                            feature_index=feature_idx,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logit[start_index],
                            end_logit=result.end_logit[end_index]))
        sentence_pred_logit = [spl for ind_spl, spl in enumerate(sentence_pred_logit) if ind_spl in example.sentence_cls]
        sp_preds = []
        new_sentence_idx = 0
        for fsm in example.full_sentences_mask:
            if fsm == 0:
                sp_preds.append(0.0)
            else:
                sp_preds.append(sentence_pred_logit[new_sentence_idx])
                new_sentence_idx += 1
        supporting_fact_preds[example.qas_id] = sp_preds
        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["start", "end", "text", "start_logit", 'end_logit'])
        seen_predictions = {}
        n_best_results = []
        for pred_result in prelim_predictions:
            feature = features[pred_result.feature_index]
            if pred_result.start_index > 0:
                get_tokens = feature.tokens[pred_result.start_index: (pred_result.end_index + 1)]
                get_tokens = [get_token for get_token in get_tokens if get_token != '[UNK]']
                origin_doc_start = example.subword2origin_index[feature.token2origin_map[pred_result.start_index]]
                origin_doc_end = example.subword2origin_index[feature.token2origin_map[pred_result.end_index]]
                origin_tokens = example.origin_tokens[origin_doc_start: origin_doc_end]
                origin_tokens = [origin_token for origin_token in origin_tokens if origin_token != '[UNK]']

                get_token_text = " ".join(get_tokens)
                get_token_text = get_token_text.replace("##", "")
                get_token_text = get_token_text.replace(" ##", "")
                get_token_text = get_token_text.strip()
                get_token_text = " ".join(get_token_text.split())

                origin_text = " ".join(origin_tokens)
                final_text = get_final_text(get_token_text, origin_text, do_lower_case, False)
                final_token_text = "".join(get_token_text.split())
                truly_final_text = "".join(final_text.split()).lower()
                start_offset = truly_final_text.find(final_token_text)
                end_offset = len(truly_final_text) - start_offset - len(final_token_text)
                if start_offset >= 0:
                    if end_offset != 0:
                        final_text = final_text[start_offset: -end_offset]
                    else:
                        final_text = final_text[start_offset:]
                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True
            else:
                # check the info
                origin_doc_start = 0
                origin_doc_end = 0
                final_text = ""
                seen_predictions[final_text] = True
            n_best_results.append(
                _NbestPrediction(
                    start=origin_doc_start,
                    end=origin_doc_end,
                    text=final_text,
                    start_logit=pred_result.start_logit,
                    end_logit=pred_result.end_logit
                )
            )
        # 是否是判断句判断
        n_best_results.append(_NbestPrediction(start=1, end=1, text="yes", start_logit=result.start_logit[1],
                                      end_logit=result.end_logit[1]))
        n_best_results.append(_NbestPrediction(start=2, end=2, text="no", start_logit=result.start_logit[2],
                                      end_logit=result.end_logit[2]))
        # 对所有结果排序
        n_best_results = sorted(n_best_results, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
        total_scores = []
        best_non_null_entry = None
        for entry in n_best_results:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)
        # check is it necessary
        nbest_json = []
        for entry_idx, entry in enumerate(n_best_results):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[entry_idx]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        all_predictions[example.qas_id] = nbest_json[0]

    return nbest_json, all_predictions, supporting_fact_preds


def run_predict(args):
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
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare model
    models = {'BertForQuestionAnsweringGraph': BertForQuestionAnsweringGraph,
              # 'BertForQuestionAnsweringForward': BertForQuestionAnsweringForward,
              'BertForQuestionAnsweringCoAttention': BertForQuestionAnsweringCoAttention,
              # 'BertForQuestionAnsweringTwoCoAttention': BertForQuestionAnsweringTwoCoAttention,
              # 'BertForQuestionAnsweringThreeCoAttention': BertForQuestionAnsweringThreeCoAttention,
              'BertForQuestionAnsweringThreeSameCoAttention': BertForQuestionAnsweringThreeSameCoAttention,
              # 'BertForQuestionAnsweringThreeSameCoAttention': BertForQuestionAnsweringThreeSameCoAttention,
              # 'BertForQuestionAnsweringThreeTransformer': BertForQuestionAnsweringThreeTransformer,
              # 'BertForQuestionAnsweringTwoTransformer': BertForQuestionAnsweringTwoTransformer,
              # 'BertForQuestionAnsweringTransformer': BertForQuestionAnsweringTransformer,
              }
    model = models[args.model_name].from_pretrained(args.checkpoints_path)
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
    dev_evaluate(args, model=model, tokenizer=tokenizer, n_gpu=n_gpu, device=device)


def dev_evaluate(args, model, tokenizer, n_gpu, device):
    dev_examples, dev_features, dev_dataloader = dev_feature_getter(args, tokenizer=tokenizer)
    model.eval()
    all_results = []
    total_loss = 0
    ans_f1 = ans_em = sp_f1 = sp_em = joint_f1 = joint_em = 0
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logit", "end_logit", "sent_logit"])
    with torch.no_grad():
        for d_step, d_batch in enumerate(tqdm(dev_dataloader, desc="Dev Interation")):
            d_example_indices = d_batch[-1]
            if n_gpu == 1:
                d_batch = tuple(
                    t.squeeze(0).to(device) for t in d_batch[:-1])  # multi-gpu does scattering it-self
            else:
                d_batch = d_batch[:-1]
            if args.dev_graph_file is not None:
                d_all_input_ids, d_all_input_mask, d_all_segment_ids, d_all_cls_mask, d_all_mask, d_all_content_len = d_batch
            else:
                d_all_input_ids, d_all_input_mask, d_all_segment_ids, d_all_cls_mask = d_batch
            # graph mask 处理
            if args.dev_graph_file:
                d_real_masks = []
                d_masks = d_all_mask.detach().cpu().tolist()
                d_content_len = d_all_content_len.detach().cpu().tolist()
                for indm, d_mask in enumerate(d_masks):
                    d_real_mask = np.zeros((args.max_seq_length, args.max_seq_length))
                    for ma in d_mask:
                        # print(ma)
                        if ma == [0, 0, 0, 0]:
                            break
                        d_real_mask[ma[0]: ma[0] + ma[-1], ma[1]:ma[1] + ma[-2]] = 1
                    # d_real_mask[0:d_content_len[indm], 0:3] = 1 - d_real_mask[0:d_content_len[indm], 0:3]
                    d_real_masks.append(d_real_mask)
                d_real_masks = torch.tensor(d_real_masks, dtype=torch.long).cuda()
            else:
                d_real_masks = None
            dev_start_logits, dev_end_logits, dev_sent_logits = model(d_all_input_ids,
                                                                      d_all_input_mask,
                                                                      d_all_segment_ids,
                                                                      sent_mask=d_all_cls_mask,
                                                                      mask=d_real_masks)

            for i, example_index in enumerate(d_example_indices):
                # start_position = start_positions[i].detach().cpu().tolist()
                # end_position = end_positions[i].detach().cpu().tolist()
                dev_start_logit = dev_start_logits[i].detach().cpu().tolist()
                dev_end_logit = dev_end_logits[i].detach().cpu().tolist()
                dev_sent_logit = dev_sent_logits[i].detach().cpu().tolist()
                dev_feature = dev_features[example_index.item()]
                unique_id = dev_feature.unique_id
                all_results.append(RawResult(unique_id=unique_id, start_logit=dev_start_logit, end_logit=dev_end_logit,
                                             sent_logit=dev_sent_logit))
    _, preds, sp_pred = write_predictions(dev_examples, dev_features, all_results)
    ans_f1, ans_em, sp_f1, sp_em, joint_f1, joint_em = evaluate(dev_examples, preds, sp_pred)
    logging("ans_f1={:4.8f},ans_em={:4.8f},sp_f1={:4.8f},sp_em={:4.8f},joint_f1={:4.8f},joint_em={:4.8f}" \
            .format(ans_f1, ans_em, sp_f1, sp_em, joint_f1, joint_em), args)
    preds = generate_preds(args.dev_file, args.dev_filter_file, dev_examples, preds, sp_pred)
    json.dump(preds, open(args.preds_file, 'w', encoding='utf-8'))
    return ans_f1, ans_em, sp_f1, sp_em, joint_f1, joint_em


def generate_preds(file_name,filter_file,eval_examples, answer_dict,sp_preds):
    results = {}
    answers = {}
    sp_facts = {}
    title_ids = {}
    data = json.load(open(file_name,'r',encoding='utf-8'))
    # data = data[:100]
    filter = json.load(open(filter_file, 'r', encoding='utf-8'))
    for d in data:
        title_id = []
        context = d['context']
        fil = filter[d['_id']]
        for indcon, con in enumerate(context):
            for i in range(len(con[1])):
                if con[1][i].strip() != '' or indcon not in fil:
                    title_id.append([con[0], i])
        title_ids[d['_id']] = title_id
    for ee in eval_examples:
        answers[ee.qas_id] = answer_dict[ee.qas_id]['text']
        sp_pred = sp_preds[ee.qas_id]
        sp_title_id = []
        title_id = title_ids[ee.qas_id]
        assert len(sp_pred) == len(title_id)
        for spp,ti in zip(sp_pred,title_id):
            if spp > 0.5:
                sp_title_id.append(ti)
        sp_facts[ee.qas_id] = sp_title_id
    results['answer'] = answers
    results['sp'] = sp_facts
    return results


def dev_feature_getter(args, tokenizer):
    dev_examples = reader_read_examples(args.dev_file,
                                        related_paragraph_file=args.dev_filter_file,
                                        tokenizer=tokenizer,
                                        is_training='dev')
    if not os.path.exists(args.feature_cache_path):
        os.makedirs(args.feature_cache_path)
    # TODO: check suffix
    dev_feature_file = '{}/reader_dev_{}_{}_{}_{}'.format(args.feature_cache_path,
                                                              list(filter(None, args.bert_model.split('/'))).pop(),
                                                              str(args.max_seq_length),
                                                              str(args.doc_stride),
                                                              str(args.feature_suffix))
    if os.path.exists(dev_feature_file) and args.use_file_cache:
        with open(dev_feature_file, "rb") as dev_f:
            dev_features = pickle.load(dev_f)
    else:
        dev_features = convert_examples_to_features(
            examples=dev_examples,
            tokenizer=tokenizer,
            graph_file=args.dev_graph_file,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            is_training='dev'
        )
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info("  Saving dev features into cached file %s", dev_feature_file)
            with open(dev_feature_file, "wb") as writer:
                pickle.dump(dev_features, writer)
    print("dev feature num: {}".format(len(dev_features)))
    d_all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
    d_all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
    d_all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
    d_all_sentence_mask = torch.tensor([f.sentence_mask for f in dev_features], dtype=torch.long)
    # TODO检查是否需要graph mask
    if args.dev_graph_file is not None:
        d_all_mask = torch.tensor([f.graph_mask for f in dev_features], dtype=torch.long)
        d_all_content_len = torch.tensor([f.content_len for f in dev_features], dtype=torch.long)
    d_all_example_index = torch.arange(d_all_input_ids.size(0), dtype=torch.long)
    if args.dev_graph_file is not None:
        dev_data = TensorDataset(d_all_input_ids, d_all_input_mask, d_all_segment_ids,
                                 d_all_sentence_mask, d_all_mask, d_all_content_len, d_all_example_index)
    else:
        dev_data = TensorDataset(d_all_input_ids, d_all_input_mask, d_all_segment_ids,
                                 d_all_sentence_mask, d_all_example_index)
    if args.local_rank == -1:
        dev_sampler = RandomSampler(dev_data)
    else:
        dev_sampler = DistributedSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.val_batch_size)
    return dev_examples, dev_features, dev_dataloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--bert_model", default='bert-base-cased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--checkpoints_path", default='../checkpoints/qa_base', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--model_name", type=str, default='BertForQuestionAnsweringThreeSameCoAttention',
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--dev_file", default='../data/hotpot_data/hotpot_dev_distractor_v1.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--dev_filter_file", default='../data/selector/second_hop_result/dev_related.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--dev_graph_file", default=None, type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--preds_file", default='../data/predictions.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--feature_cache_path", default="../data/reader_predict", type=str,
                        help="feature cache path")
    parser.add_argument("--feature_suffix", default='three_same_coattention', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--use_file_cache", default=True, type=bool,
                        help="use file cache or not")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=256, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--val_batch_size", default=32, type=int, help="Total batch size for validation.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--output_log", type=str, default='../log/reader_predict.log', )
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
    run_predict(args)
