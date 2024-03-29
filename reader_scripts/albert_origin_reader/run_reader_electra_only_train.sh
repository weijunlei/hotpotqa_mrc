#!/bin/bash
echo "----------------------------------------------------"
# albert-xxlarge-v2
export CUDA_VISIBLE_DEVICES="1"
BERT_MODEL=google/electra-large-discriminator
MODEL_NAME=ElectraForQuestionAnsweringQANet

# truly train setting
TRAIN_DIR=../../data/checkpoints/20220118_electra_test
TRAIN_TRAIN_FILE=../../data/hotpot_data/hotpot_train_labeled_data_v3.json
TRAIN_DEV_FILE=../../data/hotpot_data/hotpot_dev_labeled_data_v3.json
TRAIN_LOG=20220118_electra_test
TRAIN_CACHE=../../data/cache/20220118_electra_test
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
cd ../../albert_src/origin_reader
python -u origin_reader_model.py \
  --bert_model $BERT_MODEL \
  --output_dir $TRAIN_DIR \
  --model_name $MODEL_NAME \
  --log_prefix  $TRAIN_LOG \
  --overwrite_result True \
  --train_file $TRAIN_TRAIN_FILE \
  --dev_file  $TRAIN_DEV_FILE \
  --train_supporting_para_file ../../data/selector/20211219_second_hop_electra_large_1e_paragraph_selector_12_result/train_related.json \
  --dev_supporting_para_file ../../data/selector/20211219_second_hop_electra_large_1e_paragraph_selector_12_result/dev_related.json \
  --feature_cache_path $TRAIN_CACHE \
  --train_batch_size 12 \
  --warmup_proportion 0.1 \
  --gradient_accumulation_steps 1 \
  --local_rank -1 \
  --learning_rate 5e-6 \
  --val_batch_size 32 \
  --save_model_step 500 \
  --num_train_epochs 3.0
echo "----------------------------------------------------"

echo "train done!"
