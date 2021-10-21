#!/bin/bash
echo "----------------------------------------------------"
echo "start train reader model"
export CUDA_VISIBLE_DEVICES="0,2"
# model choice BertForQuestionAnsweringCoAttention,
               # BertForQuestionAnsweringThreeCoAttention,
               # BertForQuestionAnsweringThreeSameCoAttention,
               # BertForQuestionAnsweringForward
               # BertForQuestionAnsweringForwardBest
               # BertSelfAttentionAndCoAttention
               # BertTransformer
               # BertSkipConnectTransformer
               # BertForQuestionAnsweringForwardWithEntity
cd ../../entity_src/origin_reader
python -u origin_reader_model.py \
  --bert_model bert-base-uncased \
  --output_dir ../../data/checkpoints/qa_base_20211021_with_entity_wo_question_entity \
  --model_name BertForQuestionAnsweringForwardWithEntity \
  --log_prefix qa_base_20211021_with_entity_wo_question_entity \
  --overwrite_result True \
  --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3_with_entity_label.json \
  --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3_with_entity_label.json \
  --train_supporting_para_file ../../data/selector/second_hop_related_paragraph_result/train_related.json \
  --dev_supporting_para_file ../../data/selector/second_hop_related_paragraph_result/dev_related.json \
  --feature_cache_path ../../data/cache/qa_base_20211021_with_entity_wo_question_entity \
  --train_batch_size 12 \
  --local_rank -1 \
  --val_batch_size 64 \
  --save_model_step 3000 \
  --num_train_epochs 5.0
echo "----------------------------------------------------"
