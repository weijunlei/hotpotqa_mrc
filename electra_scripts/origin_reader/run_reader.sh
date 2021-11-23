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
cd ../../electra_src/origin_reader
python -u origin_reader_model.py \
  --bert_model google/electra-large-discriminator \
  --output_dir ../../data/checkpoints/20211123_sent_loss15_electra_large_5e \
  --model_name ElectraForQuestionAnsweringForwardBest \
  --log_prefix 20211123_sent_loss15_electra_large_5e \
  --overwrite_result True \
  --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3_with_entity_label.json \
  --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3_with_entity_label.json \
  --train_supporting_para_file ../../data/selector/second_hop_related_paragraph_result/train_related.json \
  --dev_supporting_para_file ../../data/selector/second_hop_related_paragraph_result/dev_related.json \
  --feature_cache_path ../../data/cache/20211123_sent_loss15_electra_large_5e \
  --train_batch_size 12 \
  --local_rank -1 \
  --learning_rate 5e-5 \
  --val_batch_size 64 \
  --save_model_step 1000 \
  --num_train_epochs 8.0
echo "----------------------------------------------------"
