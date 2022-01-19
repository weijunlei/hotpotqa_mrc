#!/bin/bash
echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="2,3"
BERT_MODEL=google/electra-large-discriminator
MODEL_NAME=ElectraForQuestionAnsweringQANet
BERT_MODEL=google/electra-large-discriminator
MODEL_NAME=ElectraForQuestionAnsweringQANet
PRETRAIN_LOG=20220119_test_no_unk_lr20_epoch5
PRETRAIN_TRAIN_FILE=../../data/hotpot_data/hotpot_labeled_data_squad.json
PRETRAIN_DEV_FILE=../../data/hotpot_data/hotpot_dev_labeled_data_v3.json
PRETRAIN_CACHE=../../data/cache/20220119_test_no_unk_lr20_epoch5
PRETRAIN_DIR=../../data/checkpoints/20220119_test_no_unk_lr20_epoch5

# truly train setting
TRAIN_DIR=../../data/checkpoints/20220119_test_no_unk_lr20_epoch5_step
TRAIN_TRAIN_FILE=../../data/hotpot_data/hotpot_train_labeled_data_v3.json
TRAIN_DEV_FILE=../../data/hotpot_data/hotpot_dev_labeled_data_v3.json
TRAIN_LOG=20220119_test_no_unk_lr20_epoch5_step
TRAIN_CACHE=../../data/cache/20220119_test_no_unk_lr20_epoch5_step
# model choice BertForQuestionAnsweringCoAttention,
               # BertForQuestionAnsweringThreeCoAttention,
               # BertForQuestionAnsweringThreeSameCoAttention,
               # BertForQuestionAnsweringForward
               # BertForQuestionAnsweringForwardBest
               # BertSelfAttentionAndCoAttention
               # BertTransformer
               # BertSkipConnectTransformer
               # BertForQuestionAnsweringForwardWithEntity
               # BertForQuestionAnsweringForwardWithEntityOneMask
               # hotpot_train_labeled_data_with_squad_with_entity_label
               # ElectraForQuestionAnsweringThreeCrossAttention
               # google/electra-large-discriminator
               # ElectraForQuestionAnsweringQANet
cd ../../electra_no_unk_src/origin_reader

python -u origin_reader_model.py \
  --bert_model $BERT_MODEL \
  --output_dir  $PRETRAIN_DIR \
  --model_name $MODEL_NAME \
  --log_prefix $PRETRAIN_LOG \
  --overwrite_result True \
  --train_file $PRETRAIN_TRAIN_FILE \
  --dev_file $PRETRAIN_DEV_FILE \
  --train_supporting_para_file ../../data/hotpot_data/train_golden.json \
  --dev_supporting_para_file ../../data/selector/20211219_second_hop_electra_large_1e_paragraph_selector_12_result/dev_related.json \
  --feature_cache_path $PRETRAIN_CACHE \
  --train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --local_rank -1 \
  --learning_rate 2e-5 \
  --val_batch_size 32 \
  --save_model_step 5000 \
  --num_train_epochs 5.0
echo "----------------------------------------------------"
echo "pretrain done!"


python -u origin_reader_model.py \
  --bert_model $BERT_MODEL \
  --output_dir $TRAIN_DIR \
  --model_name $MODEL_NAME \
  --checkpoint_path $PRETRAIN_DIR \
  --log_prefix  $TRAIN_LOG \
  --overwrite_result True \
  --train_file $TRAIN_TRAIN_FILE \
  --dev_file  $TRAIN_DEV_FILE \
  --train_supporting_para_file ../../data/hotpot_data/train_golden.json \
  --dev_supporting_para_file ../../data/selector/20211219_second_hop_electra_large_1e_paragraph_selector_12_result/dev_related.json \
  --feature_cache_path $TRAIN_CACHE \
  --train_batch_size 16 \
  --warmup_proportion 0.1 \
  --gradient_accumulation_steps 1 \
  --local_rank -1 \
  --learning_rate 2e-5 \
  --val_batch_size 32 \
  --save_model_step 500 \
  --num_train_epochs 5.0
echo "----------------------------------------------------"

echo "train done!"
