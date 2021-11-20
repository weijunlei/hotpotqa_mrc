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
from second_select_predict_config import get_config

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
    best_paragraph_file = os.path.join(args.first_predict_result_path, args.best_paragraph_file)
    dev_examples = read_hotpotqa_examples(input_file=args.predict_file,
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

    dev_features = convert_examples_to_features(
        examples=dev_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        is_training='dev'
    )
    logger.info("dev feature num: {}".format(len(dev_features)))
    dev_data = LazyLoadTensorDataset(features=dev_features, is_training=False)
    dev_sampler = RandomSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.val_batch_size)
    return dev_examples, dev_dataloader, dev_features


def dev_predict(args,
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
    write_predictions(args, dev_features, dev_examples, all_results)


def write_predictions(args, features, examples, all_result):
    """ 写入结果 """
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
    dev_data = json.load(open(args.predict_file, "r"))
    format_result = {}
    best_paragraph_file = os.path.join(args.first_predict_result_path, args.best_paragraph_file)
    first_predict_best_dict = json.load(open(best_paragraph_file, "r"))
    error_num = 0
    best_two_paragraph_dict = {}
    all_paragraph_dict = {}
    for info in dev_data:
        q_id = info['_id']
        format_result[q_id] = [[0] * 10, [0] * 10]
        best_two_paragraph_dict[q_id] = []
        all_paragraph_dict[q_id] = [0] * 10
    for k, v in paragraph_results.items():
        q_id, context_id = k.split('_')
        try:
            context_id = int(context_id)
            if q_id not in format_result:
                format_result[q_id] = [[0] * 10, [0] * 10]
            format_result[q_id][0][context_id] = v
            all_paragraph_dict[q_id][context_id] = v
        except Exception as e:
            import pdb; pdb.set_trace()
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
        best_two_paragraph_dict[k].append(first_best_idx)
        for context_idx, (predict_result, true_result) in enumerate(zip(predict_results, true_results)):
            if predict_result == max_predict_result and not has_result:
                if true_result == 1:
                    has_result = True
                    best_two_paragraph_dict[k].append(context_idx)
    if not os.path.exists(args.predict_result_path):
        os.makedirs(args.predict_result_path)
    best_relate_paragraph_file = os.path.join(args.predict_result_path, args.best_relate_paragraph_file)
    all_paragraph_file = os.path.join(args.predict_result_path, args.all_paragraph_file)
    json.dump(best_two_paragraph_dict, open(best_relate_paragraph_file, "w", encoding="utf-8"))
    json.dump(all_paragraph_dict, open(all_paragraph_file, "w", encoding="utf-8"))


def run_predict(args, rank=0, world_size=1):
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
    if not args.overwrite_result and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory () already exists and is not empty.")
    if rank == 0 and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = RobertaTokenizer.from_pretrained(args.bert_model,
                                                 do_lower_case=args.do_lower_case)

    model = model_dict[args.model_name].from_pretrained(args.checkpoint_path)
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
    dev_examples, dev_dataloader, dev_features = get_dev_data(args, tokenizer=tokenizer)
    dev_predict(args,
                model,
                dev_dataloader,
                n_gpu,
                device,
                dev_features,
                tokenizer,
                dev_examples)



if __name__ == '__main__':
    parser = get_config()
    args = parser.parse_args()
    if not args.use_ddp:
        run_predict(args)
    else:
        processes = []
        for rank in range(args.world_size):
            p = Process(target=run_predict, args=(args, rank, args.world_size))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
