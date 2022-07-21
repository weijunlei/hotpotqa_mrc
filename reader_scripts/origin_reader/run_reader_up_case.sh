#!/bin/bash
echo "----------------------------------------------------"
echo "start to sleep"
date
sleep 3.5h
echo "end sleep and start to run model!"
date
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
cd ../../stable_src/origin_reader
python -u origin_reader_model.py \
  --bert_model google/electra-large-discriminator \
  --output_dir ../../data/checkpoints/20211229_qa_net_with_20_weight_epoch_3_up_case \
  --checkpoint_path ../../data/checkpoints/20211225_electra_large_dynamic_weight_bs12_pre_trained \
  --model_name ElectraForQuestionAnsweringQANet \
  --log_prefix 20211229_qa_net_with_20_weight_epoch_3_up_case \
  --do_lower_case False \
  --overwrite_result True \
  --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
  --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
  --train_supporting_para_file ../../data/hotpot_data/train_golden.json \
  --dev_supporting_para_file ../../data/selector/20211217_second_hop_electra_base_just_paragraph_selector_12_value_setting_result/dev_related.json \
  --feature_cache_path ../../data/cache/20211229_qa_net_with_20_weight_epoch_3_up_case \
  --train_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --local_rank -1 \
  --learning_rate 2e-5 \
  --warmup_proportion 0.1 \
  --val_batch_size 32 \
  --save_model_step 500 \
  --num_train_epochs 3.0
echo "----------------------------------------------------"


python -u origin_reader_model.py \
  --bert_model google/electra-large-discriminator \
  --output_dir ../../data/checkpoints/20211229_qa_net_with_20_weight_epoch_4_up_case \
  --checkpoint_path ../../data/checkpoints/20211225_electra_large_dynamic_weight_bs12_pre_trained \
  --model_name ElectraForQuestionAnsweringQANet \
  --log_prefix 20211229_qa_net_with_20_weight_epoch_4_up_case \
  --do_lower_case False \
  --overwrite_result True \
  --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
  --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
  --train_supporting_para_file ../../data/hotpot_data/train_golden.json \
  --dev_supporting_para_file ../../data/selector/20211217_second_hop_electra_base_just_paragraph_selector_12_value_setting_result/dev_related.json \
  --feature_cache_path ../../data/cache/20211229_qa_net_with_20_weight_epoch_4_up_case \
  --train_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --local_rank -1 \
  --learning_rate 2e-5 \
  --warmup_proportion 0.075 \
  --val_batch_size 32 \
  --save_model_step 500 \
  --num_train_epochs 4.0
echo "----------------------------------------------------"


python -u origin_reader_model.py \
  --bert_model google/electra-large-discriminator \
  --output_dir ../../data/checkpoints/20211229_qa_net_with_20_weight_epoch_5_up_case \
  --checkpoint_path ../../data/checkpoints/20211225_electra_large_dynamic_weight_bs12_pre_trained \
  --model_name ElectraForQuestionAnsweringQANet \
  --log_prefix 20211229_qa_net_with_20_weight_epoch_5_up_case \
  --do_lower_case False \
  --overwrite_result True \
  --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
  --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
  --train_supporting_para_file ../../data/hotpot_data/train_golden.json \
  --dev_supporting_para_file ../../data/selector/20211217_second_hop_electra_base_just_paragraph_selector_12_value_setting_result/dev_related.json \
  --feature_cache_path ../../data/cache/20211229_qa_net_with_20_weight_epoch_5_up_case \
  --train_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --local_rank -1 \
  --learning_rate 2e-5 \
  --warmup_proportion 0.06 \
  --val_batch_size 32 \
  --save_model_step 500 \
  --num_train_epochs 5.0
echo "----------------------------------------------------"
