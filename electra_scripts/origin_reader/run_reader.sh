#!/bin/bash
echo "----------------------------------------------------"
echo "start train reader model"
export CUDA_VISIBLE_DEVICES="1,3"
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
cd ../../electra_src/origin_reader
python -u origin_reader_model.py \
  --bert_model google/electra-large-discriminator \
  --output_dir ../../data/checkpoints/20211209_sent_loss30_electra_large_cross_attention \
  --model_name ElectraForQuestionAnsweringCrossAttention \
  --log_prefix 20211209_sent_loss30_electra_large_cross_attention \
  --overwrite_result True \
  --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3_with_entity_label.json \
  --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3_with_entity_label.json \
  --train_supporting_para_file ../../data/hotpot_data/train_golden.json \
  --dev_supporting_para_file ../../data/selector/second_hop_related_paragraph_result/dev_related.json \
  --feature_cache_path ../../data/cache/20211208_sent_loss20_electra_large_cross_attention_wi_dp \
  --train_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --local_rank 0 \
  --learning_rate 3e-5 \
  --val_batch_size 64 \
  --save_model_step 1000 \
  --num_train_epochs 3.0
echo "----------------------------------------------------"
