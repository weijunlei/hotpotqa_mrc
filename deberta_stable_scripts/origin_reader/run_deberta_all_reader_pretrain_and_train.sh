#!/bin/bash
echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="2,0"
BERT_MODEL=microsoft/deberta-v3-large
MODEL_NAME=DebertaForQuestionAnsweringQANet
PRETRAIN_LOG=20211231_deberta_qanet
PRETRAIN_TRAIN_FILE=../../data/hotpot_data/hotpot_labeled_data_squad.json
PRETRAIN_DEV_FILE=../../data/hotpot_data/hotpot_dev_labeled_data_v3.json
PRETRAIN_CACHE=../../data/cache/20211231_deberta_qanet
PRETRAIN_DIR=../../data/checkpoints/20211231_deberta_qanet

# truly train setting
TRAIN_DIR=../../data/checkpoints/20211231_deberta_qanet_step
TRAIN_TRAIN_FILE=../../data/hotpot_data/hotpot_train_labeled_data_v3.json
TRAIN_DEV_FILE=../../data/hotpot_data/hotpot_dev_labeled_data_v3.json
TRAIN_LOG=20211231_deberta_qanet_step
TRAIN_CACHE=../../data/cache/20211231_deberta_qanet_step
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
cd ../../deberta_stable_src/origin_reader
python -u origin_reader_model.py \
  --bert_model $BERT_MODEL \
  --output_dir  $PRETRAIN_DIR \
  --model_name $MODEL_NAME \
  --log_prefix $PRETRAIN_LOG \
  --overwrite_result True \
  --train_file $PRETRAIN_TRAIN_FILE \
  --dev_file $PRETRAIN_DEV_FILE \
  --train_supporting_para_file ../../data/hotpot_data/train_golden.json \
  --dev_supporting_para_file ../../data/selector/20211217_second_hop_electra_base_just_paragraph_selector_12_value_setting_result/dev_related.json \
  --feature_cache_path $PRETRAIN_CACHE \
  --train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --local_rank -1 \
  --learning_rate 1e-5 \
  --val_batch_size 32 \
  --save_model_step 5000 \
  --num_train_epochs 3.0
echo "----------------------------------------------------"
echo "pretrain done!"


python -u origin_reader_model.py \
  --bert_model $BERT_MODEL \
  --output_dir $TRAIN_DIR \
  --checkpoint_path $PRETRAIN_DIR \
  --model_name $MODEL_NAME \
  --log_prefix  $TRAIN_LOG \
  --overwrite_result True \
  --train_file $TRAIN_TRAIN_FILE \
  --dev_file  $TRAIN_DEV_FILE \
  --train_supporting_para_file ../../data/hotpot_data/train_golden.json \
  --dev_supporting_para_file ../../data/selector/20211217_second_hop_electra_base_just_paragraph_selector_12_value_setting_result/dev_related.json \
  --feature_cache_path $TRAIN_CACHE \
  --train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --local_rank -1 \
  --learning_rate 1e-5 \
  --val_batch_size 32 \
  --save_model_step 500 \
  --num_train_epochs 3.0
echo "----------------------------------------------------"

echo "train done!"
