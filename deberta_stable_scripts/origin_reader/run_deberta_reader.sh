#!/bin/bash
echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="0"

# model choice BertForQuestionAnsweringCoAttention,
               # BertForQuestionAnsweringThreeCoAttention,
               # BertForQuestionAnsweringThreeSameCoAttention,
               # BertForQuestionAnsweringForward
               # BertForQuestionAnsweringForwardBest
               # BertSelfAttentionAndCoAttention
               # BertTransformer
               #  --checkpoint_path ../../data/checkpoints/20211225_electra_large_dynamic_weight_bs12_pre_trained \
               # BertSkipConnectTransformer
               # BertForQuestionAnsweringForwardWithEntity
               # BertForQuestionAnsweringForwardWithEntityOneMask
               # hotpot_train_labeled_data_with_squad_with_entity_label
               # ElectraForQuestionAnsweringThreeCrossAttention
               # google/electra-large-discriminator
               # ElectraForQuestionAnsweringQANet
cd ../../deberta_stable_src/origin_reader
python -u origin_reader_model.py \
  --bert_model microsoft/deberta-v3-large \
  --output_dir ../../data/checkpoints/20211230_deberta \
  --model_name DebertaForQuestionAnsweringQANet \
  --log_prefix 20211230_deberta \
  --overwrite_result True \
  --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
  --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
  --train_supporting_para_file ../../data/hotpot_data/train_golden.json \
  --dev_supporting_para_file ../../data/selector/20211217_second_hop_electra_base_just_paragraph_selector_12_value_setting_result/dev_related.json \
  --feature_cache_path ../../data/cache/20211230_deberta \
  --train_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --local_rank -1 \
  --learning_rate 2e-5 \
  --val_batch_size 32 \
  --save_model_step 500 \
  --num_train_epochs 3.0
echo "----------------------------------------------------"
