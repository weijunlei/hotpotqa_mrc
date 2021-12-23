#!/bin/bash
echo "----------------------------------------------------"
echo "start train reader model"
export CUDA_VISIBLE_DEVICES="0,1"
# bert-base-uncased
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
python -u origin_reader_model.py \
  --bert_model google/electra-base-discriminator \
  --output_dir ../../data/checkpoints/20211222_electra_base_gelu_act_with_weight \
  --model_name BertForQuestionAnsweringQANetAttentionWeight \
  --log_prefix 20211222_electra_base_gelu_act_with_weight \
  --overwrite_result True \
  --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3_with_entity_label.json \
  --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3_with_entity_label.json \
  --train_supporting_para_file ../../data/hotpot_data/train_golden.json \
  --dev_supporting_para_file ../../data/selector/20211217_second_hop_electra_base_just_paragraph_selector_12_value_setting_result/dev_related.json \
  --feature_cache_path ../../data/cache/20211222_electra_base_gelu_act_with_weight \
  --train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --local_rank -1 \
  --learning_rate 2e-5 \
  --val_batch_size 128 \
  --save_model_step 1000 \
  --num_train_epochs 3.0
echo "----------------------------------------------------"