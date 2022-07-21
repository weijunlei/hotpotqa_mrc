#!/bin/bash
echo "----------------------------------------------------"
echo "start to sleep"
date
sleep 7h
echo "sleep done and start to train"
date
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# model choice BertForQuestionAnsweringCoAttention,
               # BertForQuestionAnsweringThreeCoAttention,
               # BertForQuestionAnsweringThreeSameCoAttention,
               # BertForQuestionAnsweringForward
               # BertForQuestionAnsweringForwardBest
               # BertSelfAttentionAndCoAttention
               # BertTransformer
               # BertSkipConnectTransformer
               # BertForQuestionAnsweringForwardWithEntity
               #  --checkpoint_path ../../data/checkpoints/20211225_electra_large_dynamic_weight_bs12_pre_trained \
               #  --checkpoint_path ../../data/checkpoints/20211225_electra_large_dynamic_weight_bs12_pre_trained \
               # BertForQuestionAnsweringForwardWithEntityOneMask
               # hotpot_train_labeled_data_with_squad_with_entity_label
               # ElectraForQuestionAnsweringThreeCrossAttention
               # google/electra-large-discriminator
               # ElectraForQuestionAnsweringQANet
               # AlbertForQuestionAnsweringQANet
cd ../../stable_src/origin_reader
python -u origin_reader_model.py \
  --bert_model google/electra-large-discriminator \
  --output_dir ../../data/checkpoints/20211228_electra_large_trained_origin_epoch10_warmup_10_electra_wo_pretrained \
  --model_name ElectraForQuestionAnsweringQANet \
  --log_prefix 20211228_electra_large_trained_origin_epoch10_warmup_10_electra_wo_pretrained \
  --overwrite_result True \
  --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
  --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
  --train_supporting_para_file ../../data/hotpot_data/train_golden.json \
  --dev_supporting_para_file ../../data/selector/20211217_second_hop_electra_base_just_paragraph_selector_12_value_setting_result/dev_related.json \
  --feature_cache_path ../../data/cache/20211228_electra_large_trained_origin_epoch10_warmup_10_electra_wo_pretrained \
  --train_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --warmup_proportion 0.1 \
  --local_rank -1 \
  --learning_rate 2e-5 \
  --val_batch_size 16 \
  --save_model_step 200 \
  --num_train_epochs 10.0
echo "----------------------------------------------------"
