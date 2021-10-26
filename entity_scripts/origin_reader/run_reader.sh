#!/bin/bash
echo "----------------------------------------------------"
echo "start train reader model"
export CUDA_VISIBLE_DEVICES="0"
# model choice BertForQuestionAnsweringCoAttention,
               # BertForQuestionAnsweringThreeCoAttention,
               # BertForQuestionAnsweringThreeSameCoAttention,
               # BertForQuestionAnsweringForward
               # BertForQuestionAnsweringForwardBest
               # BertSelfAttentionAndCoAttention
               # BertTransformer
               # BertSkipConnectTransformer
#                BertForQuestionAnsweringForwardWithEntity
               # BertForQuestionAnsweringForwardWithEntityOneMask
cd ../../entity_src/origin_reader
python -u origin_reader_model.py \
  --bert_model bert-base-uncased \
  --output_dir ../../data/checkpoints/20211026_naive_bert_best_with_entity_embedding_with_transformer \
  --model_name BertForQuestionAnsweringForwardWithEntityTransformer \
  --log_prefix 20211026_naive_bert_best_with_entity_embedding_with_transformer \
  --overwrite_result True \
  --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3_with_entity_label.json \
  --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3_with_entity_label.json \
  --train_supporting_para_file ../../data/selector/second_hop_related_paragraph_result/train_related.json \
  --dev_supporting_para_file ../../data/selector/second_hop_related_paragraph_result/dev_related.json \
  --feature_cache_path ../../data/cache/qa_base_20211023_with_entity_wi_question_entity \
  --train_batch_size 12 \
  --local_rank -1 \
  --val_batch_size 64 \
  --save_model_step 3000 \
  --num_train_epochs 5.0
echo "----------------------------------------------------"
