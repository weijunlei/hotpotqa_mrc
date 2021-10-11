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

from origin_reader_helper import HotpotQAReaderExample, \
    HotpotQAInputFeatures, \
    cut_sent, \
    fix_span, \
    find_nearest, \
    _improve_answer_span, \
    _check_is_max_context, \
    get_final_text, \
    _get_best_indexes, \
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


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
                if feature.sentence_mask[sentence_logit_idx] == 1 and feature.token_is_max_context.get(
                        sentence_logit_idx, False):
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
        sentence_pred_logit = [spl for ind_spl, spl in enumerate(sentence_pred_logit) if
                               ind_spl in example.sentence_cls]
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
        for i, entry in enumerate(n_best_results):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        all_predictions[example.qas_id] = nbest_json[0]

    return nbest_json, all_predictions, supporting_fact_preds


def convert_examples2file(train_examples,
                          start_idxs,
                          end_idxs,
                          cached_train_features_file,
                          tokenizer):
    """ 将实例转化为特征并存储在文件中"""
    total_feature_num = 0
    for start_idx in range(len(start_idxs)):
        logger.info("start example idx: {} all num:{}".format(start_idx, len(start_idxs)))
        truly_train_examples = train_examples[start_idxs[start_idx]: end_idxs[start_idx]]
        new_train_cache_file = cached_train_features_file + '_' + str(start_idx)
        if os.path.exists(new_train_cache_file) and args.use_file_cache:
            with open(new_train_cache_file, "rb") as f:
                train_features = pickle.load(f)
        else:
            logger.info("convert {} example(s) to features...".format(len(truly_train_examples)))
            train_features = convert_examples_to_features(
                examples=truly_train_examples,
                tokenizer=tokenizer,
                graph_file=args.train_graph_file,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                is_training='train')
            logger.info("features gotten!")
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving train features into cached file %s", cached_train_features_file)
                logger.info("start saving features...")
                with open(cached_train_features_file + '_' + str(start_idx), "wb") as writer:
                    pickle.dump(train_features, writer)
                logger.info("saving features done!")
        total_feature_num += len(train_features)
    logger.info('train feature_num: {}'.format(total_feature_num))
    return total_feature_num


def train_iterator(args,
                   start_idxs,
                   cached_train_features_file,
                   tokenizer,
                   n_gpu,
                   model,
                   device,
                   optimizer,
                   num_train_optimization_steps):
    best_predict_f1 = 0
    train_loss = 0
    global_steps = 0
    for epoch_idx in trange(int(args.num_train_epochs), desc="Epoch"):
        for start_idx in trange(len(start_idxs), desc="Data"):
            with open(cached_train_features_file + "_" + str(start_idx), "rb") as reader:
                train_features = pickle.load(reader)
            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_start_position = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
            all_end_position = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
            all_sent_mask = torch.tensor([f.sentence_mask for f in train_features], dtype=torch.long)
            all_sent_lbs = torch.tensor([f.sentence_labels for f in train_features], dtype=torch.long)
            all_sent_weight = torch.tensor([f.sentence_weight for f in train_features], dtype=torch.float)
            # TODO: check the mask result
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_position,
                                       all_end_position,
                                       all_sent_mask, all_sent_lbs, all_sent_weight)
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_data)
            else:
                train_sampler = DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if n_gpu == 1:
                    batch = tuple(t.squeeze(0).to(device) for t in batch)  # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, start_position, end_position, sent_mask, sent_lbs, sent_weight = batch
                real_masks = None

                loss, _, _, _ = model(input_ids, input_mask, segment_ids, start_positions=start_position,
                                      end_positions=end_position,
                                      sent_mask=sent_mask, sent_lbs=sent_lbs, sent_weight=sent_weight, mask=real_masks)
                if n_gpu > 1:
                    loss = loss.sum()  # mean() to average on multi-gpu.
                logger.info(" step = %d, train_loss=%f", global_steps, loss)
                train_loss += loss
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if (global_steps + 1) % 100 == 0 and (step + 1) % args.gradient_accumulation_steps == 0:
                    logger.info("epoch:{:3d},data:{:3d},global_step:{:8d},loss:{:8.3f}".format(epoch_idx, start_idx,
                                                                                               global_steps,
                                                                                               train_loss))
                    train_loss = 0
                if (global_steps + 1) % args.save_model_step == 0 and (
                        step + 1) % args.gradient_accumulation_steps == 0:
                    all_results = []
                    total_loss = 0
                    # start dev evaluate
                    ans_f1, ans_em, sp_f1, sp_em, joint_f1, joint_em = dev_evaluate(args,
                                                                                    model,
                                                                                    tokenizer=tokenizer,
                                                                                    n_gpu=n_gpu,
                                                                                    device=device)
                    logger.info(
                        "epoch={:3d}, data={:3d},step = {:6d},ans_f1={:4.8f},ans_em={:4.8f},sp_f1={:4.8f},sp_em={:4.8f},joint_f1={:4.8f},joint_em={:4.8f},best_predict_f1={:4.8f},total_loss={:8.3f}" \
                            .format(epoch_idx, start_idx, global_steps, ans_f1, ans_em, sp_f1, sp_em, joint_f1,
                                    joint_em, best_predict_f1,
                                    total_loss))
                    logger.info(
                        "ans_f1: {},ans_em: {},sp_f1: {},sp_em: {},joint_f1: {},joint_em: {}".format(ans_f1, ans_em,
                                                                                                     sp_f1, sp_em,
                                                                                                     joint_f1,
                                                                                                     joint_em))
                    if joint_f1 > best_predict_f1:
                        best_predict_f1 = joint_f1
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(args.output_dir, 'pytorch_model.bin')
                        torch.save(model_to_save.state_dict(), output_model_file)
                        output_config_file = os.path.join(args.output_dir, 'config.json')
                        with open(output_config_file, 'w') as f:
                            f.write(model_to_save.config.to_json_string())
                        logger.info('saving model')
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used and handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_steps / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_steps += 1
            del train_features, all_input_ids, all_input_mask, all_segment_ids, all_start_position, all_end_position, all_sent_lbs, all_sent_mask, all_sent_weight, train_data, train_dataloader
            gc.collect()
    # 模型最后评估
    model.eval()
    all_results = []
    total_loss = 0
    # start dev evaluate
    ans_f1, ans_em, sp_f1, sp_em, joint_f1, joint_em = dev_evaluate(args,
                                                                    model,
                                                                    tokenizer=tokenizer,
                                                                    n_gpu=n_gpu,
                                                                    device=device)
    logger.info(
        "step = {:6d},ans_f1={:4.8f},ans_em={:4.8f},sp_f1={:4.8f},sp_em={:4.8f},joint_f1={:4.8f},joint_em={:4.8f},best_predict_f1={:4.8f},total_loss={:8.3f}" \
            .format(global_steps, ans_f1, ans_em, sp_f1, sp_em, joint_f1, joint_em, best_predict_f1,
                    total_loss))
    if joint_f1 >= best_predict_f1:
        best_predict_f1 = joint_f1
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, 'pytorch_model.bin')
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, 'config.json')
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
        logger.info('saving model')


def run_train(args):
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

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None

    # Prepare model
    models_dict = {
        # 'BertForQuestionAnsweringGraph': BertForQuestionAnsweringGraph,
        'BertForQuestionAnsweringThreeSameCoAttention': BertForQuestionAnsweringThreeSameCoAttention,
        'BertForQuestionAnsweringCoAttention': BertForQuestionAnsweringCoAttention, }
    model = models_dict[args.model_name].from_pretrained(args.bert_model)

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

    if not os.path.exists(args.feature_cache_path):
        os.makedirs(args.feature_cache_path)
    cached_train_features_file = args.feature_cache_path + '/_{0}_{1}_{2}_{3}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride),
        args.feature_suffix)
    model.train()
    train_examples = reader_read_examples(
        input_file=args.train_file, related_paragraph_file=args.train_filter_file, tokenizer=tokenizer,
        is_training='train')
    example_num = len(train_examples)
    logger.info('train example_num: {}'.format(example_num))
    max_train_data_size = 100000
    start_idxs = list(range(0, example_num, max_train_data_size))
    end_idxs = [x + max_train_data_size for x in start_idxs]
    end_idxs[-1] = example_num
    logger.info('{} examples and {} example file(s)'.format(example_num, len(start_idxs)))
    # TODO: check random
    # 缓存或者重新读取数据
    random.shuffle(train_examples)
    total_feature_num = convert_examples2file(train_examples,
                                              start_idxs=start_idxs,
                                              end_idxs=end_idxs,
                                              cached_train_features_file=cached_train_features_file,
                                              tokenizer=tokenizer)

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
    train_iterator(args,
                   start_idxs,
                   cached_train_features_file,
                   tokenizer,
                   n_gpu,
                   model,
                   device,
                   optimizer,
                   num_train_optimization_steps)


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
            d_example_indices = d_batch[-1].squeeze()
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
                        if ma == [0, 0, 0, 0]:
                            break
                        d_real_mask[ma[0]: ma[0] + ma[-1], ma[1]:ma[1] + ma[-2]] = 1
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
    # add model train
    model.train()
    return ans_f1, ans_em, sp_f1, sp_em, joint_f1, joint_em


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
    logger.info("dev feature num: {}".format(len(dev_features)))
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
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default='../../data/checkpoints/qa_base_20210923_coattention', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--model_name", type=str, default='BertForQuestionAnsweringCoAttention',
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--train_file", default='../../data/hotpot_data/hotpot_train_labeled_data_v3.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--dev_file", default='../../data/hotpot_data/hotpot_dev_distractor_v1.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--train_filter_file", default='../../data/selector/second_hop_related_paragraph_result/train_related.json',
                        type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--dev_filter_file", default='../../data/selector/second_hop_related_paragraph_result/dev_related.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--use_file_cache", dest='use_file_cache', action='store_true', help="use cache file or not")
    parser.add_argument("--feature_cache_path", default="../../data/cache/qa_base_20210923_coattention")
    parser.add_argument("--train_graph_file", default=None, type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--dev_graph_file", default=None, type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--feature_suffix", default='graph3oqsur', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=256, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--val_batch_size", default=128, type=int, help="Total batch size for validation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=2.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--output_log", type=str, default='../log/reader_model.txt', )
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
    run_train(args=args)
