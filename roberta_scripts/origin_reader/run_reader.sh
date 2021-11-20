#!/bin/bash
echo "----------------------------------------------------"
echo "start train reader model"
export CUDA_VISIBLE_DEVICES="0,1,2"
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
cd ../../roberta_src/origin_reader
python -u origin_reader_model.py \
  --bert_model roberta-large \
  --output_dir ../../data/checkpoints/  \
  --model_name RobertaForQuestionAnsweringForwardBest \
  --log_prefix 20211113_roberta_forward_best_epoch5_unk_bs24 \
  --overwrite_result True \
  --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3_with_entity_label.json \
  --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3_with_entity_label.json \
  --train_supporting_para_file ../../data/selector/second_hop_related_paragraph_result/train_related.json \
  --dev_supporting_para_file ../../data/selector/second_hop_related_paragraph_result/dev_related.json \
  --feature_cache_path ../../data/cache/20211113_roberta_forward_best_epoch5_unk_bs24 \
  --train_batch_size 24 \
  --local_rank -1 \
  --learning_rate 1e-5 \
  --val_batch_size 64 \
  --save_model_step 1000 \
  --num_train_epochs 8.0
echo "----------------------------------------------------"
