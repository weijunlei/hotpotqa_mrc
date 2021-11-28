import argparse
import json

import torch
import random
import numpy as np
import os
import gc
import sys
import logging
import pickle
import collections
from tqdm import trange, tqdm
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (ElectraTokenizer,
                          ElectraForSequenceClassification,
                          get_linear_schedule_with_warmup,
                          BertTokenizer,
                          BertForSequenceClassification,
                          RobertaTokenizer,
                          RobertaForSequenceClassification)
from transformers import AdamW
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from first_hop_helper import read_hotpotqa_examples, convert_example_to_features
from first_selector_config import get_config



# 日志设置
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


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def evaluate(args, model, eval_dataloader, features, tokenizer, prefix='dev', step=0):
    """ 答案评估 """
    eval_result = {}
    try:
        predict_result = {}
        dev_data = json.load(open(args.dev_file, "r"))
        softmax = torch.nn.Softmax(dim=1)
        true_example_result = {}
        all_true_num = 0
        for info in dev_data:
            context = info['context']
            support_facts = info['supporting_facts']
            support_facts_set = set(['{}_{}'.format(x[0], x[1]) for x in support_facts])
            qa_id = info['_id']
            true_example_result[qa_id] = [0 for _ in range(10)]
            for paragraph_idx, paragraph in enumerate(context):
                title, sentences = paragraph
                related = False
                for sentence_idx, sentence in enumerate(sentences):
                    if '{}_{}'.format(title, sentence_idx) in support_facts_set:
                        related = True
                if related:
                    true_example_result[qa_id][paragraph_idx] = 1
                    all_true_num += 1
        for batch in tqdm(eval_dataloader, desc="Evaluating..."):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                if 'roberta' in args.bert_model.lower():
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                    }
                else:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                    }
                example_indices = batch[3]
                outputs = model(**inputs)
                outputs = outputs.logits
                outputs = softmax(outputs)
            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = eval_feature.unique_id
                qas_id, feature_id = unique_id.rsplit('_', 1)
                origin_predict_result = predict_result.get(qas_id, 0)
                logits = to_list(outputs[i])
                new_prob = 0
                if len(logits) == 2:
                    new_prob = logits[1]
                else:
                    new_prob = logits[0]
                if new_prob >= origin_predict_result:
                    predict_result[qas_id] = new_prob
        example_result = {}
        write_result = {}
        for qas_id, qas_prob in predict_result.items():
            qa_id, paragraph_idx = qas_id.rsplit('_', 1)
            paragraph_idx = int(paragraph_idx)
            if qa_id not in example_result:
                example_result[qa_id] = [0 for _ in range(10)]
            example_result[qa_id][paragraph_idx] = qas_prob

        best_acc_num = 0
        best_precision_num = 0
        recall_num = 0
        for qa_id, example_info in example_result.items():
            max_thread = max(example_info)
            true_info = true_example_result[qa_id]
            is_first = True
            for paragraph_idx, paragraph_prob in enumerate(example_info):
                if is_first and paragraph_prob == max_thread:
                    write_result[qa_id] = paragraph_idx
                    if true_info[paragraph_idx] == 1:
                        best_acc_num += 1
                        best_precision_num += 1
                        recall_num += 1
                        is_first = False
                        example_info[paragraph_idx] = 0
                        max_thread = max(example_info)
                elif not is_first and paragraph_prob == max_thread:
                    if true_info[paragraph_idx] == 1:
                        recall_num += 1
        eval_result['acc'] = 1.0 * best_acc_num / len(example_result)
        eval_result['best_precision_num'] = 1.0 * best_precision_num / len(example_result)
        eval_result['recall'] = 1.0 * recall_num / all_true_num
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        write_result_file = "{}/{}_eval_result.json".format(args.output_dir, step)
        with open(write_result_file, "w") as f:
            json.dump(write_result, f)
    except Exception as e:
        import pdb; pdb.set_trace()
    return eval_result



def run_train(rank=0, world_size=1):
    """ 模型训练 """
    global logger
    parser = get_config()
    args = parser.parse_args()
    # 配置日志文件
    if rank == 0 and not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    log_path = os.path.join(args.log_path, 'log_selector_{}_{}_{}_{}_{}_{}.log'.format(args.log_prefix,
                                                                              args.bert_model.split('/')[-1],
                                                                              args.output_dir.split('/')[-1],
                                                                              args.train_batch_size,
                                                                              args.max_seq_length,
                                                                              args.doc_stride))
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(args.summary)
    logger = logger_config(log_path=log_path, log_prefix='')
    logger.info('-' * 15 + '所有配置' + '-' * 15)
    logger.info("所有参数配置如下：")
    for arg in vars(args):
        logger.info('{}: {}'.format(arg, getattr(args, arg)))
    logger.info('-' * 30)
    # 分布式或多卡训练
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
        args.n_gpu = n_gpu
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        args.n_gpu = n_gpu
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    args.device = device
    # 随机种子设定
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    # 梯度积累不小于1
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    if not args.train_file:
        raise ValueError('`train_file` is not specified!')
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_result:
        raise ValueError('output_dir {} already exists!'.format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
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
    # TODO check do lower case work?
    # 设置分词器和模型
    cls_token = '[CLS]'
    sep_token = '[SEP]'
    unk_token = '[UNK]'
    pad_token = '[PAD]'
    if 'roberta' in args.bert_model.lower():
        tokenizer = RobertaTokenizer.from_pretrained(args.bert_model)
        model = RobertaForSequenceClassification.from_pretrained(args.bert_model)
        cls_token = '<s>'
        sep_token = '</s>'
        unk_token = '<unk>'
        pad_token = '<pad>'
    elif 'bert' in args.bert_model.lower():
        tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        model = BertForSequenceClassification.from_pretrained(args.bert_model)
    elif 'electra' in args.bert_model.lower():
        tokenizer = ElectraTokenizer.from_pretrained(args.bert_model)
        model = ElectraForSequenceClassification.from_pretrained(args.bert_model)
    train_examples = None
    num_train_optimization_steps = None


    # 开始读取训练数据
    logger.info("start read example...")
    if rank == 0 and not os.path.exists(args.feature_cache_path):
        os.makedirs(args.feature_cache_path)

    logger.info("start read features...")
    if rank == 0 and not os.path.exists(args.feature_cache_path):
        os.makedirs(args.feature_cache_path)
    cached_train_features_file = '{}/train_feature_file_{}_{}_{}_{}'.format(args.feature_cache_path,
                                                                      args.bert_model.split('/')[-1],
                                                                      str(args.max_seq_length),
                                                                      str(args.doc_stride),
                                                                      args.feature_suffix)
    if os.path.exists(cached_train_features_file):
        with open(cached_train_features_file, "rb") as reader:
            train_features = pickle.load(reader)
    else:
        train_examples = read_hotpotqa_examples(input_file=args.train_file,
                                                is_training='train')
        train_features = convert_example_to_features(examples=train_examples,
                                                     tokenizer=tokenizer,
                                                     max_seq_length=args.max_seq_length,
                                                     is_training='train',
                                                     cls_token=cls_token,
                                                     sep_token=sep_token,
                                                     unk_token=unk_token,
                                                     pad_token=pad_token
                                                     )
        with open(cached_train_features_file, "wb") as writer:
            pickle.dump(train_features, writer)
    cached_dev_features_file = '{}/dev_test4_feature_file_{}_{}_{}_{}'.format(args.feature_cache_path,
                                                                            args.bert_model.split('/')[-1],
                                                                            str(args.max_seq_length),
                                                                            str(args.doc_stride),
                                                                            args.feature_suffix)
    logger.info("load train feature done!")
    if os.path.exists(cached_dev_features_file):
        with open(cached_dev_features_file, "rb") as reader:
            dev_features = pickle.load(reader)
    else:
        logger.info("start read dev examples...")
        dev_examples = read_hotpotqa_examples(input_file=args.dev_file,
                                              is_training='dev')
        logger.info("start read dev features...")
        dev_features = convert_example_to_features(examples=dev_examples,
                                                   tokenizer=tokenizer,
                                                   max_seq_length=args.max_seq_length,
                                                   is_training='dev',
                                                   cls_token='<s>',
                                                   sep_token='</s>',
                                                   unk_token='<unk>',
                                                   pad_token='<pad>')
        with open(cached_dev_features_file, "wb") as writer:
            pickle.dump(dev_features, writer)
    logger.info("load dev feature done!")
    total_feature_num = len(train_features)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # 展开数据
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    train_labels = torch.tensor([f.is_related for f in train_features], dtype=torch.long)
    train_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, train_labels)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    # 展开验证集数据
    dev_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
    dev_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
    d_all_example_index = torch.arange(dev_input_ids.size(0), dtype=torch.long)
    dev_labels = torch.tensor([f.is_related for f in dev_features], dtype=torch.long)
    dev_dataset = TensorDataset(dev_input_ids, dev_input_mask, all_segment_ids, d_all_example_index, dev_labels)
    if args.local_rank == -1:
        dev_sampler = SequentialSampler(dev_dataset)
    else:
        dev_sampler = DistributedSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.val_batch_size)

    # 参数配置
    logger.info("parameter setting...")
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    warmup_steps = int(t_total * args.warmup_proportion)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.output_dir, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.output_dir, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.output_dir, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.output_dir, "scheduler.pt")))
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
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
    global_step = 1
    epochs_trained = 0
    save_steps = args.save_model_step
    logging_steps = args.logging_steps
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.output_dir):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.output_dir.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            steps_trained_in_current_epoch *= args.gradient_accumulation_steps
        except ValueError:
            logger.info("  Starting fine-tuning.")
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if 'roberta' in args.bert_model.lower():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    # "token_type_ids": batch[2],
                    "labels": batch[3]
                }
            else:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3]
                }
            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and logging_steps > 0 and global_step % logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, dev_dataloader, dev_features, tokenizer, prefix='dev', step=global_step)
                        for key, value in results.items():
                            tb_writer.add_scalar("squad_eval_{}".format(key), value, global_step)
                            logger.info("evaluate result: global_step: {} key: {} value: {}".format(global_step, key, value))
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and save_steps > 0 and global_step % save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break


if __name__ == '__main__':
    use_ddp = False
    if not use_ddp:
        run_train()
    else:
        world_size = 2
        processes = []
        for rank in range(world_size):
            p = Process(target=run_train, args=(rank, world_size))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()




