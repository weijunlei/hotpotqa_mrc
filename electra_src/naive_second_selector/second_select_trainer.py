from __future__ import absolute_import, division, print_function
import collections
import json
import logging
import os
import random
import sys
from io import open
import torch
import numpy as np
from multiprocessing import Process
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import pickle
import gc
from transformers import RobertaTokenizer

from roberta_model import RobertaForParagraphClassification
from second_select_helper import read_hotpotqa_examples, convert_examples_to_features
from lazy_dataloader import LazyLoadTensorDataset
from second_select_config import get_config

sys.path.append("../pretrain_model")
from optimization import BertAdam, warmup_linear

model_dict = {'RobertaForParagraphClassification': RobertaForParagraphClassification}

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
logger = None


def logger_config(log_path, log_prefix='lwj', write2console=True):
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


def get_dev_data(args, tokenizer):
    global logger
    best_paragraph_file = os.path.join(args.first_predict_result_path, args.best_dev_paragraph_file)
    dev_examples = read_hotpotqa_examples(input_file=args.dev_file,
                                          best_paragraph_file=best_paragraph_file,
                                          tokenizer=tokenizer,
                                          is_training='dev')
    logger.info("dev examples: {}".format(len(dev_examples)))
    cached_dev_features_file = '{}/naive_second_selector_dev_feature_file_{}_{}_{}_{}'.format(args.feature_cache_path,
                                                                                              args.bert_model.split(
                                                                                                  '/')[-1],
                                                                                              str(args.max_seq_length),
                                                                                              str(args.doc_stride),
                                                                                              args.log_prefix)

    if os.path.exists(cached_dev_features_file):
        with open(cached_dev_features_file, "rb") as f:
            dev_features = pickle.load(f)
    else:
        dev_features = convert_examples_to_features(
            examples=dev_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            is_training='dev'
        )
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info(" Saving dev features into cached file %s", cached_dev_features_file)
            with open(cached_dev_features_file, "wb") as writer:
                pickle.dump(dev_features, writer)
    logger.info("dev feature num: {}".format(len(dev_features)))
    dev_data = LazyLoadTensorDataset(features=dev_features, is_training=False)
    dev_sampler = RandomSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.val_batch_size)
    return dev_examples, dev_dataloader, dev_features


def get_train_data(args, tokenizer):
    """ 获取训练数据 """
    global logger
    cached_train_features_file = '{}/naive_second_selector_train_feature_file_{}_{}_{}_{}'.format(
        args.feature_cache_path,
        args.bert_model.split('/')[-1],
        str(args.max_seq_length),
        str(args.doc_stride),
        args.log_prefix)
    logger.info("reading example from file...")
    best_paragraph_file = os.path.join(args.first_predict_result_path, args.best_train_paragraph_file)
    train_examples = read_hotpotqa_examples(input_file=args.train_file,
                                            best_paragraph_file=best_paragraph_file,
                                            tokenizer=tokenizer,
                                            is_training='train')
    random.shuffle(train_examples)
    example_num = len(train_examples)
    logger.info("train example num: {}".format(example_num))
    max_train_num = 100000
    start_idxs = list(range(0, example_num, max_train_num))
    end_idxs = [x + max_train_num for x in start_idxs]
    end_idxs[-1] = example_num
    total_feature_num = 0
    for idx in range(len(start_idxs)):
        new_cache_file = cached_train_features_file + '_' + str(idx)
        if os.path.exists(new_cache_file):
            logger.info("start reading feature from cache file: {}...".format(new_cache_file))
            with open(new_cache_file, "rb") as f:
                train_features = pickle.load(f)
            logger.info("read features done!")
        else:
            logger.info("start reading features from origin examples...")
            tmp_examples = train_examples[start_idxs[idx]: end_idxs[idx]]
            train_features = convert_examples_to_features(
                examples=tmp_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                is_training='train'
            )
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("Saving train features into cache file {}".format(new_cache_file))
                with open(new_cache_file, "wb") as writer:
                    pickle.dump(train_features, writer)
                logger.info("Saving features to file: {} done!".format(new_cache_file))
        total_feature_num += len(train_features)
        del train_features
        gc.collect()
    logger.info("train feature num: {}".format(total_feature_num))
    return total_feature_num, start_idxs, cached_train_features_file


def dev_evaluate(args,
                 model,
                 dev_dataloader,
                 n_gpu,
                 device,
                 dev_features,
                 tokenizer,
                 dev_examples):
    """ 验证结果 """
    acc = pre = recall = f1 = em = 0
    model.eval()
    all_results = []
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "logit"])
    total_loss = 0
    with torch.no_grad():
        for step, d_batch in enumerate(tqdm(dev_dataloader, desc="Predict Iteration")):
            try:
                example_indices = d_batch[-1]
                if n_gpu == 1:
                    d_batch = tuple(x.squeeze(0).to(device) for x in d_batch[:-1])
                else:
                    d_batch = d_batch[:-1]
                input_ids, input_mask, segment_ids = d_batch
                dev_logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
                dev_logits = torch.sigmoid(dev_logits)
                for i, example_index in enumerate(example_indices):
                    dev_logit = dev_logits[i].detach().cpu().tolist()
                    dev_logit.reverse()
                    dev_feature = dev_features[example_index.item()]
                    unique_id = dev_feature.unique_id
                    all_results.append(RawResult(unique_id=unique_id,
                                                 logit=dev_logit))
            except Exception as e:
                import pdb; pdb.set_trace()
    model.train()
    acc, pre, recall, f1, em = write_predictions(args, dev_features, dev_examples, all_results)
    return acc, pre, recall, f1, em


def write_predictions(args, features, examples, all_result):
    """ 写入结果 """
    acc = pre = recall = f1 = em = 0
    example_index2features = collections.defaultdict(list)
    for feature in features:
        example_index2features[feature.example_id].append(feature)
    unique_id2result = {x[0]: x for x in all_result}
    paragraph_results = {}
    for example_index, example in enumerate(examples):
        features = example_index2features[example_index]
        qas_id = '_'.join(features[0].unique_id.split('_')[:-1])
        if len(features) == 0:
            get_feature = feature[0]
            get_feature_id = get_feature.unique_id
            raw_result = unique_id2result[get_feature_id].logit
            paragraph_results[qas_id] = raw_result[0]
        else:
            for get_feature in features:
                get_feature_id = get_feature.unique_id
                raw_result = unique_id2result[get_feature_id].logit
                if qas_id not in paragraph_results:
                    paragraph_results[qas_id] = raw_result[0]
                else:
                    paragraph_results[qas_id] = max(paragraph_results[qas_id], raw_result[0])
    return evaluate_result(args, paragraph_results)


def evaluate_result(args, paragraph_results, thread=0.5):
    dev_data = json.load(open(args.dev_file, "r"))
    acc = pre = recall = f1 = em = 0
    format_result = {}
    best_paragraph_file = os.path.join(args.first_predict_result_path, args.best_dev_paragraph_file)
    first_predict_best_dict = json.load(open(best_paragraph_file, "r"))
    error_num = 0
    for info in dev_data:
        q_id = info['_id']
        format_result[q_id] = [[0] * 10, [0] * 10]
    for k, v in paragraph_results.items():
        q_id, context_id = k.split('_')
        context_id = int(context_id)
        if q_id not in format_result:
            format_result[q_id] = [[0] * 10, [0] * 10]
        format_result[q_id][0][context_id] = v
    for info in dev_data:
        q_id = info['_id']
        context = info['context']
        supporting_facts = info["supporting_facts"]
        # title$index
        supporting_facts_set = set(["{}${}".format(x[0], x[1]) for x in supporting_facts])
        for paragraph_idx, paragraph in enumerate(context):
            title, sentences = paragraph
            for sent_idx, sent in enumerate(sentences):
                if "{}${}".format(title, sent_idx) in supporting_facts_set:
                    format_result[q_id][1][paragraph_idx] = 1
    all_true_num = 0
    for k, v in format_result.items():
        predict_results, true_results = v
        max_predict_result = max(predict_results)
        true_num = 0
        has_result = False
        first_best_idx = first_predict_best_dict[k]
        for context_idx, (predict_result, true_result) in enumerate(zip(predict_results, true_results)):
            if predict_result == max_predict_result and not has_result:
                if true_result == 1:
                    has_result = True
                    acc += 1
                    true_num += 1
            if true_result == 1:
                all_true_num += 1
                if first_best_idx == context_idx:
                    true_num += 1
                    acc += 1
        recall += true_num if true_num <= 2 else 2
        em += 1 if true_num == 2 else 0
    predict_num = len(format_result) * 2
    recall = 1.0 * acc / all_true_num
    acc = 1.0 * acc / predict_num
    pre = acc
    f1 = 2 * (acc * recall) / (acc + recall)
    em = 1.0 * em / len(format_result)
    return acc, pre, recall, f1, em


def run_train(args, rank=0, world_size=1):
    # 配置日志文件
    global logger
    if rank == 0 and not os.path.exists(args.log_path):
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
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        logger.info("start train on nccl!")
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
    if rank == 0 and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = RobertaTokenizer.from_pretrained(args.bert_model,
                                                 do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    model = model_dict[args.model_name].from_pretrained(args.bert_model)
    # 半精度和并行化使用设置
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        logger.info("setting model {}..".format(rank))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
        logger.info("setting model {} done!".format(rank))
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # 参数配置
    logger.info("parameter setting...")
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    global_step = 0
    logger.info("start read example...")
    # 获取训练集数据
    if rank == 0 and not os.path.exists(args.feature_cache_path):
        os.makedirs(args.feature_cache_path)
    total_feature_num, start_idxs, cached_train_features_file = get_train_data(args=args, tokenizer=tokenizer)
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
        logger.info("t_total: {}".format(num_train_optimization_steps))
    max_f1 = 0
    print_loss = 0
    for epoch_idx in trange(int(args.num_train_epochs), desc="Epoch"):
        for ind in trange(len(start_idxs), desc='Data'):
            new_cache_file = cached_train_features_file + '_' + str(ind)
            logger.info("loading file: {}".format(new_cache_file))
            with open(new_cache_file, "rb") as reader:
                train_features = pickle.load(reader)
            logger.info("load file: {} done!".format(new_cache_file))
            train_data = LazyLoadTensorDataset(features=train_features, is_training=True)
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_data)
            else:
                train_sampler = DistributedSampler(train_data)

            train_dataloader = DataLoader(train_data,
                                          sampler=train_sampler,
                                          batch_size=args.train_batch_size)
            for step, batch in enumerate(tqdm(train_dataloader,
                                              desc="Data:{}/{} Epoch:{}/{} Iteration".format(ind,
                                                                                             len(start_idxs),
                                                                                             epoch_idx,
                                                                                             args.num_train_epochs))):
                if n_gpu == 1:
                    batch = tuple(t.squeeze(0).to(device) for t in batch)  # multi-gpu does scattering it-self
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, example_indexs, labels = batch
                if len(input_ids.shape) < 2:
                    input_ids = input_ids.unsqueeze(0)
                    segment_ids = segment_ids.unsqueeze(0)
                    input_mask = input_mask.unsqueeze(0)
                    if labels is not None and len(labels.shape) < 2:
                        labels = labels.unsqueeze(0)
                loss, _ = model(input_ids=input_ids,
                                attention_mask=input_mask,
                                token_type_ids=segment_ids,
                                label=labels)
                if n_gpu > 1:
                    loss = loss.sum()  # mean() to average on multi-gpu.
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
                    if rank == 0:
                        dev_examples, dev_dataloader, dev_features = get_dev_data(args,
                                                                                  tokenizer=tokenizer)
                        acc, pre, recall, f1, em = dev_evaluate(args,
                                                                model,
                                                                dev_dataloader,
                                                                n_gpu,
                                                                device,
                                                                dev_features,
                                                                tokenizer,
                                                                dev_examples)
                        logger.info("epoch:{} data: {} step: {}".format(epoch_idx, ind, global_step))
                        logger.info("acc:{} pre:{} recall:{} em: {}".format(
                            acc, pre, recall, em
                        ))
                        del dev_examples, dev_dataloader, dev_features
                        gc.collect()
                        logger.info("max_f1: {}".format(max_f1))
                        if f1 > max_f1:
                            logger.info("get better model in step: {} with f1: {}".format(global_step, f1))
                            max_f1 = f1
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Only save the model it-self
                            output_model_file = os.path.join(args.output_dir, 'pytorch_model_best.bin')
                            # output_model_file = os.path.join(args.output_dir, 'pytorch_model_{}.bin'.format(global_step))
                            torch.save(model_to_save.state_dict(), output_model_file)
                            output_config_file = os.path.join(args.output_dir, 'config.json')
                            with open(output_config_file, 'w') as f:
                                f.write(model_to_save.config.to_json_string())
                            logger.info('saving step: {} model'.format(global_step))
            # 内存清除
            del train_features, input_ids, input_mask, segment_ids
            del labels, train_data, train_dataloader
            gc.collect()
    # 保存最后的模型
    logger.info("t_total: {} global steps: {}".format(num_train_optimization_steps, global_step))
    if rank == 0:
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, 'pytorch_model.bin')
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, 'config.json')
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
        logger.info('saving step: {} model'.format(global_step))


if __name__ == '__main__':
    parser = get_config()
    args = parser.parse_args()
    if not args.use_ddp:
        run_train(args)
    else:
        processes = []
        for rank in range(args.world_size):
            p = Process(target=run_train, args=(args, rank, args.world_size))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
