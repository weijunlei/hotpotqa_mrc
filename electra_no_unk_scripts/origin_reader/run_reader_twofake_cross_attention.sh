#!/bin/bash
echo "----------------------------------------------------"
echo "start to sleep"
date
sleep 6h
date
echo "sleep end and start to train"
export CUDA_VISIBLE_DEVICES="3,2"
BERT_MODEL=google/electra-large-discriminator


# truly train setting
MODEL_NAME=ElectraForQuestionAnsweringTwoFakeCrossAttention
TRAIN_DIR=../../data/checkpoints/20220105_two_fake_cross_attention
TRAIN_TRAIN_FILE=../../data/hotpot_data/hotpot_train_labeled_data_v3.json
TRAIN_DEV_FILE=../../data/hotpot_data/hotpot_dev_labeled_data_v3.json
TRAIN_LOG=20220105_two_fake_cross_attention
TRAIN_CACHE=../../data/cache/20220105_two_fake_cross_attention
python -u origin_reader_model.py \
  --bert_model $BERT_MODEL \
  --output_dir $TRAIN_DIR \
  --model_name $MODEL_NAME \
  --log_prefix  $TRAIN_LOG \
  --overwrite_result True \
  --train_file $TRAIN_TRAIN_FILE \
  --dev_file  $TRAIN_DEV_FILE \
  --train_supporting_para_file ../../data/hotpot_data/train_golden.json \
  --dev_supporting_para_file ../../data/selector/20211217_second_hop_electra_base_just_paragraph_selector_12_value_setting_result/dev_related.json \
  --feature_cache_path $TRAIN_CACHE \
  --train_batch_size 16 \
  --warmup_proportion 0.1 \
  --gradient_accumulation_steps 1 \
  --local_rank -1 \
  --learning_rate 1.5e-5 \
  --val_batch_size 32 \
  --save_model_step 500 \
  --num_train_epochs 3.0
echo "----------------------------------------------------"

echo "train done!"