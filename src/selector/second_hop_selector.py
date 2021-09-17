import argparse
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
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Sampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from second_hop_data_helper import read_second_hotpotqa_examples, convert_examples_to_second_features
from second_hop_prediction_helper import write_predictions
sys.path.append("../pretrain_model")
from modeling_bert import *
from optimization import BertAdam, warmup_linear
from tokenization import BertTokenizer

# 日志设置
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def dev_evaluate(args, model, tokenizer, n_gpu, device):
    dev_examples, dev_features, dev_dataloader = dev_feature_getter(args, tokenizer=tokenizer)
    model.eval()
    all_results = []
    total_loss = 0
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "logit"])

    with torch.no_grad():
        for d_step, d_batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            d_example_indices = d_batch[-1]
            if n_gpu == 1:
                d_batch = tuple(
                    t.squeeze(0).to(device) for t in d_batch[:-1])  # multi-gpu does scattering it-self
            d_all_input_ids, d_all_input_mask, d_all_segment_ids, d_all_cls_mask, d_all_cls_label, d_all_cls_weight = d_batch[
                                                                                                                      :-1]
            dev_loss, dev_logits = model(d_all_input_ids, d_all_input_mask, d_all_segment_ids,
                                         cls_mask=d_all_cls_mask, cls_label=d_all_cls_label,
                                         cls_weight=d_all_cls_weight)
            dev_loss = torch.sum(dev_loss)
            dev_logits = torch.sigmoid(dev_logits)
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

    acc, prec, em, rec = write_predictions(args, dev_examples, dev_features, all_results)
    # pickle.dump(all_results, open('all_results.pkl', 'wb'))
    model.train()
    del dev_examples, dev_features, dev_dataloader
    gc.collect()
    return acc, prec, em, rec, total_loss


def dev_feature_getter(args, tokenizer):
    dev_examples = read_second_hotpotqa_examples(args.dev_file,
                                                 best_paragraph_file="{}/{}".format(args.first_predict_result_path,
                                                                                    args.dev_best_paragraph_file),
                                                 related_paragraph_file="{}/{}".format(args.first_predict_result_path,
                                                                                       args.dev_related_paragraph_file),
                                                 new_context_file="{}/{}".format(args.first_predict_result_path,
                                                                                 args.dev_new_context_file),

                                                 is_training='dev')
    if not os.path.exists(args.feature_cache_path):
        os.makedirs(args.feature_cache_path)
    dev_feature_file = '{}/selector_2_dev_{}_{}_{}'.format(args.feature_cache_path,
                                                              list(filter(None, args.bert_model.split('/'))).pop(),
                                                              str(args.max_seq_length),
                                                              str(args.sent_overlap))
    if os.path.exists(dev_feature_file) and args.use_file_cache:
        with open(dev_feature_file, "rb") as dev_f:
            dev_features = pickle.load(dev_f)
    else:
        dev_features = convert_examples_to_second_features(
            examples=dev_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
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
    return dev_examples, dev_features, dev_dataloader


def convert_example2file(examples,
                         start_idxs,
                         end_idxs,
                         cached_train_features_file,
                         tokenizer):
    """ 将example转化为feature并存储在文件中 """
    total_feature_num = 0
    for idx in range(len(start_idxs)):
        logger.info("start example idx: {} all num: {}".format(idx, len(start_idxs)))
        truly_train_examples = examples[start_idxs[idx]: end_idxs[idx]]
        new_train_cache_file = cached_train_features_file + '_' + str(idx)
        if os.path.exists(new_train_cache_file) and args.use_file_cache:
            with open(new_train_cache_file, "rb") as f:
                train_features = pickle.load(f)
        else:
            logger.info("convert {} example(s) to features...".format(len(truly_train_examples)))
            train_features = convert_examples_to_second_features(
                examples=truly_train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                is_training='train')
            logger.info("features gotten!")
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving train features into cached file {}".format(cached_train_features_file))
                logger.info("start saving features...")
                with open(cached_train_features_file + '_' + str(idx), "wb") as writer:
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
    """ 训练 """
    best_predict_acc = 0
    train_loss = 0
    global_steps = 0
    train_features = None
    for epoch_idx in trange(int(args.num_train_epochs), desc="Epoch"):
        for start_idx in trange(len(start_idxs), desc='Data'):
            with open(cached_train_features_file + '_' + str(start_idx), "rb") as reader:
                train_features = pickle.load(reader)
            # 展开数据
            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_cls_mask = torch.tensor([f.cls_mask for f in train_features], dtype=torch.long)
            all_cls_label = torch.tensor([f.cls_label for f in train_features], dtype=torch.long)
            all_cls_weight = torch.tensor([f.cls_weight for f in train_features], dtype=torch.float)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                       all_cls_mask, all_cls_label, all_cls_weight)
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_data)
            else:
                train_sampler = DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if n_gpu == 1:
                    batch = tuple(t.squeeze(0).to(device) for t in batch)  # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, cls_mask, cls_label, cls_weight = batch

                loss, _ = model(input_ids, input_mask, segment_ids, cls_mask=cls_mask, cls_label=cls_label,
                                cls_weight=cls_weight)
                if n_gpu > 1:
                    loss = loss.sum()  # mean() to average on multi-gpu.
                logger.info("step = %d, train_loss=%f", global_steps, loss)
                train_loss += loss
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if (global_steps + 1) % 100 == 0 and (step + 1) % args.gradient_accumulation_steps == 0:
                    logger.info(
                        "epoch:{:3d},data:{:3d},global_step:{:8d},loss:{:8.3f}".format(epoch_idx, start_idx, global_step, train_loss))
                    train_loss = 0
                if (global_steps + 1) % args.save_model_step == 0 and (step + 1) % args.gradient_accumulation_steps == 0:
                    acc, prec, em, rec, total_loss = dev_evaluate(args=args,
                                                                  model=model,
                                                                  tokenizer=tokenizer,
                                                                  n_gpu=n_gpu,
                                                                  device=device)
                    logger.info("epoch: {} data idx: {} step: {}".format(epoch_idx, start_idx, global_steps))
                    logger.info(
                        "acc: {} precision: {} em: {} recall: {} total loss: {}".format(acc, prec, em, rec, total_loss))

                    if acc > best_predict_acc:
                        best_predict_acc = acc
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
            del train_features, all_input_ids, all_input_mask, all_segment_ids, all_cls_label, all_cls_mask, all_cls_weight, train_data, train_dataloader
            gc.collect()
    acc, prec, em, rec, total_loss = dev_evaluate(args=args, model=model, tokenizer=tokenizer, n_gpu=n_gpu, device=device)
    logger.info("epoch: {} data idx: {} step: {}".format(epoch_idx, start_idx, global_steps))
    logger.info("acc: {} precision: {} em: {} recall: {} total loss: {}".format(acc, prec, em, rec, total_loss))

    if acc > best_predict_acc:
        best_predict_acc = acc
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, 'pytorch_model.bin')
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, 'config.json')
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
        logger.info('saving model')


def run_train(args):
    """ train second selector """
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
    # 梯度积累不小于1
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    # 随机种子设定
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.train_file:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.over_write_result:
        raise ValueError("Output directory {} already exists and is not empty." + args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    models_dict = {"BertForRelatedSentence": BertForRelatedSentence,
                   "BertForParagraphClassification": BertForParagraphClassification}
    model = models_dict[args.model_name].from_pretrained(args.bert_model)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model == DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    param_optimizer = list(model.named_parameters())
    # 把池化去除不进行参数更新
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    global_steps = 0
    if not os.path.exists(args.feature_cache_path):
        os.makedirs(args.feature_cache_path)
    cached_train_features_file = "{}/selector_second_train_{}_{}_{}".format(args.feature_cache_path,
                                                                 list(filter(None, args.bert_model.split('/'))).pop(),
                                                                 str(args.max_seq_length),
                                                                 str(args.sent_overlap))
    train_features = None
    model.train()
    if not os.path.exists(args.first_predict_result_path):
        raise ValueError("first hop result not be predicted! " + args.first_predict_result_path)

    train_examples = read_second_hotpotqa_examples(
                                        input_file=args.train_file,
                                        best_paragraph_file="{}/{}".format(args.first_predict_result_path,
                                                                             args.best_paragraph_file),
                                        related_paragraph_file="{}/{}".format(args.first_predict_result_path,
                                                                             args.related_paragraph_file),
                                        new_context_file="{}/{}".format(args.first_predict_result_path,
                                                                        args.new_context_file),
                                        is_training='train')

    example_num = len(train_examples)
    max_train_data_size = 100000
    start_idxs = list(range(0, example_num, max_train_data_size))
    end_idxs = [x + max_train_data_size for x in start_idxs]
    end_idxs[-1] = example_num
    logger.info('{} examples and {} example file(s)'.format(example_num, start_idxs));
    random.shuffle(train_examples)
    total_feature_num = convert_example2file(examples=train_examples,
                                             start_idxs=start_idxs,
                                             end_idxs=end_idxs,
                                             cached_train_features_file=cached_train_features_file,
                                             tokenizer=tokenizer)
    logger.info("total feature num: {}".format(total_feature_num))
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
    train_iterator(args=args,
                   start_idxs=start_idxs,
                   cached_train_features_file=cached_train_features_file,
                   tokenizer=tokenizer,
                   n_gpu=n_gpu,
                   model=model,
                   device=device,
                   optimizer=optimizer,
                   num_train_optimization_steps=num_train_optimization_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 必须参数
    # 模型参数
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese."
                        )
    parser.add_argument("--over_write_result", default=True, type=bool,
                        help="over write the result")
    parser.add_argument("--output_dir", default='../checkpoints/selector/second_hop_selector', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--feature_cache_path", default="../data/cache/selector/second_hop_selector")
    parser.add_argument("--model_name", type=str, default='BertForRelated',
                        help="The output directory where the model checkpoints and predictions will be written.")
    # 数据输入
    parser.add_argument("--train_file", default='../data/hotpot_data/hotpot_train_labeled_data.json', type=str,
                        help="train_file")
    parser.add_argument("--first_predict_result_path", default="../data/selector/first_hop_result/", type=str,
                        help="The output directory of all result")
    parser.add_argument("--best_paragraph_file", default='train_best_paragraph.json', type=str,
                        help="best_paragraph_file")
    parser.add_argument("--related_paragraph_file", default='train_related_paragraph.json', type=str,
                        help="related_paragraph_file")
    parser.add_argument("--new_context_file", default='train_new_context.json', type=str,
                        help="new_context_file")
    parser.add_argument("--dev_best_paragraph_file", default='dev_best_paragraph.json', type=str,
                        help="best_paragraph_file")
    parser.add_argument("--dev_related_paragraph_file", default='dev_related_paragraph.json', type=str,
                        help="related_paragraph_file")
    parser.add_argument("--dev_new_context_file", default='dev_new_context.json', type=str,
                        help="new_context_file")
    parser.add_argument("--dev_file", default='../data/hotpot_data/hotpot_dev_labeled_data.json', type=str,
                        help="SQuAD json for evaluation. ")
    # 其他参数
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--use_file_cache", default=True, type=bool,
                        help="use the feature cache or not")
    parser.add_argument("--sent_overlap", default=2, type=int,
                        help="When splitting up a long document into chunks, "
                             "how much sentences is overlapped between chunks.")
    parser.add_argument("--train_batch_size", default=24, type=int, help="Total batch size for training.")
    parser.add_argument("--val_batch_size", default=128, type=int, help="Total batch size for validation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--output_log", type=str, default='../log/selector_2_base_2e-5.txt', )
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
                        type=int, default=5000,
                        help="The proportion of the validation set")
    args = parser.parse_args()
    run_train(args)