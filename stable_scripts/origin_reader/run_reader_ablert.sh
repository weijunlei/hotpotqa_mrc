#!/bin/bash
echo "----------------------------------------------------"
echo "start train reader model"
export CUDA_VISIBLE_DEVICES="2"

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
               # albert-xxlarge-v2
               # ElectraForQuestionAnsweringQANet
cd ../../electra_src/origin_reader
python -u origin_reader_model.py \
  --bert_model albert-xxlarge-v2 \
  --output_dir ../../data/checkpoints/20211219_sent_loss_20_2e_albert_large_qa_net_weighted_wo_bi_with_co_cross_with_electra_selector \
  --model_name ElectraForQuestionAnsweringQANet \
  --log_prefix 20211219_sent_loss_20_2e_albert_large_qa_net_weighted_wo_bi_with_co_cross_with_electra_selector \
  --overwrite_result True \
  --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3_with_entity_label.json \
  --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3_with_entity_label.json \
  --train_supporting_para_file ../../data/hotpot_data/train_golden.json \
  --dev_supporting_para_file ../../data/selector/20211217_second_hop_electra_base_just_paragraph_selector_12_value_setting_result/dev_related.json \
  --feature_cache_path ../../data/cache/20211219_sent_loss_20_2e_albert_large_qa_net_weighted_wo_bi_with_co_cross_with_electra_selector \
  --train_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --local_rank -1 \
  --learning_rate 2e-5 \
  --val_batch_size 128 \
  --save_model_step 1000 \
  --num_train_epochs 3.0
echo "----------------------------------------------------"
