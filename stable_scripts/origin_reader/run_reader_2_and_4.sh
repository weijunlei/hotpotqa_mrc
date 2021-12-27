#!/bin/bash
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
               # ElectraForQuestionAnsweringQANet
cd ../../stable_src/origin_reader
#python -u origin_reader_model.py \
#  --bert_model google/electra-large-discriminator \
#  --output_dir ../../data/checkpoints/20211226_cross_attention_selector_electra_large_attention_weight_epoch_2 \
#  --model_name ElectraForQuestionAnsweringQANetAttentionWeight \
#  --log_prefix 20211226_cross_attention_selector_electra_large_attention_weight_epoch_2 \
#  --overwrite_result True \
#  --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
#  --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
#  --train_supporting_para_file ../../data/hotpot_data/train_golden.json \
#  --dev_supporting_para_file ../../data/selector/20211222_second_hop_electra_cross_attention_1e_paragraph_selector_12_result/dev_related.json \
#  --feature_cache_path ../../data/cache/20211226_cross_attention_selector_electra_large_attention_weight_epoch_2 \
#  --train_batch_size 12 \
#  --gradient_accumulation_steps 1 \
#  --local_rank -1 \
#  --learning_rate 2e-5 \
#  --val_batch_size 128 \
#  --save_model_step 500 \
#  --num_train_epochs 2.0
#echo "----------------------------------------------------"



python -u origin_reader_model.py \
  --bert_model google/electra-large-discriminator \
  --output_dir ../../data/checkpoints/20211226_cross_attention_selector_electra_large_attention_weight_epoch_4 \
  --model_name ElectraForQuestionAnsweringQANetAttentionWeight \
  --log_prefix 20211226_cross_attention_selector_electra_large_attention_weight_epoch_4 \
  --overwrite_result True \
  --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
  --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
  --train_supporting_para_file ../../data/hotpot_data/train_golden.json \
  --dev_supporting_para_file ../../data/selector/20211222_second_hop_electra_cross_attention_1e_paragraph_selector_12_result/dev_related.json \
  --feature_cache_path ../../data/cache/20211226_cross_attention_selector_electra_large_attention_weight_epoch_4 \
  --train_batch_size 12 \
  --warmup_proportion 0.05 \
  --gradient_accumulation_steps 1 \
  --local_rank -1 \
  --learning_rate 2e-5 \
  --val_batch_size 128 \
  --save_model_step 500 \
  --num_train_epochs 4.0
echo "----------------------------------------------------"