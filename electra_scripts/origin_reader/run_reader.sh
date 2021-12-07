#!/bin/bash
echo "----------------------------------------------------"
echo "start train reader model"
export CUDA_VISIBLE_DEVICES="1"
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
cd ../../electra_src/origin_reader
python -u origin_reader_model.py \
  --bert_model google/electra-large-discriminator \
  --output_dir ../../data/checkpoints/20211207_sent_loss20_electra_large_forwardbest_3e_train_golden \
  --model_name ElectraForQuestionAnsweringForwardBest \
  --log_prefix 20211207_sent_loss20_electra_large_forwardbest_3e_train_golden \
  --overwrite_result True \
  --train_file ../../data/hotpot_data/hotpot_train_labeled_data_with_squad_with_entity_label.json \
  --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3_with_entity_label.json \
  --train_supporting_para_file ../../data/hotpot_data/train_golden.json \
  --dev_supporting_para_file ../../data/selector/second_hop_related_paragraph_result/dev_related.json \
  --feature_cache_path ../../data/cache/20211207_sent_loss20_electra_large_forwardbest_3e_train_golden \
  --train_batch_size 12 \
  --local_rank -1 \
  --learning_rate 3e-5 \
  --val_batch_size 64 \
  --save_model_step 1000 \
  --num_train_epochs 3.0
echo "----------------------------------------------------"
