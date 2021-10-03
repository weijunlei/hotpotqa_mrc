#!/bin/bash
echo "----------------------------------------------------"
echo "start train reader model"
export CUDA_VISIBLE_DEVICES="0,1"
# model choice BertForQuestionAnsweringCoAttention,
               # BertForQuestionAnsweringThreeCoAttention,
               # BertForQuestionAnsweringThreeSameCoAttention,
               # BertForQuestionAnsweringForward
               # BertForQuestionAnsweringForwardBest
               # BertSelfAttentionAndCoAttention
cd ../../src/origin_reader
python -u origin_reader_model.py \
  --bert_model bert-base-uncased \
  --output_dir ../../data/checkpoints/qa_base_20211003_just_paragraph_coattention \
  --model_name BertForQuestionAnsweringCoAttention \
  --log_prefix qa_base_20211003_just_paragraph_coattention \
  --overwrite_result True \
  --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
  --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
  --train_filter_file ../../data/selector/second_hop_just_paragraph_result/train_related.json \
  --dev_filter_file ../../data/selector/second_hop_just_paragraph_result/dev_related.json \
  --feature_cache_path ../../data/cache/qa_base_20211001_just_paragraph_v2 \
  --train_batch_size 12 \
  --val_batch_size 64 \
  --save_model_step 3000 \
  --num_train_epochs 5.0
echo "----------------------------------------------------"