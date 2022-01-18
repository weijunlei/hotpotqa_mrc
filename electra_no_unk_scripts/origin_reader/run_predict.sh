#!/bin/bash
echo "----------------------------------------------------"
echo "start train reader model"
export CUDA_VISIBLE_DEVICES="0"
CHECKPOINT_PATH=/data1/jptang1/leiwj/data_transform/bs12_squad_model/model
OUTPUT_DIR=/data1/jptang1/leiwj/data_transform/bs12_squad_model/model
TEST_PARAGRAPH=../../data/selector/20211217_second_hop_electra_base_just_paragraph_selector_12_value_setting_result/dev_related.json

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
cd ../../stable_src/origin_reader
python -u origin_reader_predictor.py \
  --bert_model google/electra-large-discriminator \
  --checkpoint_dir $CHECKPOINT_PATH \
  --output_dir $OUTPUT_DIR \
  --predict_file reader_predict.json \
  --model_name ElectraForQuestionAnsweringQANet \
  --log_prefix 20211221_electra_predict_test \
  --overwrite_result True \
  --test_file ../../data/hotpot_data/hotpot_dev_distractor_v1.json \
  --test_supporting_para_file $TEST_PARAGRAPH \
  --feature_cache_path ../../data/cache/reader_test \
  --gradient_accumulation_steps 1 \
  --local_rank -1 \
  --val_batch_size 12
echo "----------------------------------------------------"
