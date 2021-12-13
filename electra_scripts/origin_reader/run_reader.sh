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
               # BertForQuestionAnsweringForwardWithEntity
               # BertForQuestionAnsweringForwardWithEntityOneMask
               # hotpot_train_labeled_data_with_squad_with_entity_label
               # ElectraForQuestionAnsweringThreeCrossAttention
               # google/electra-large-discriminator
cd ../../electra_src/origin_reader
python -u origin_reader_model.py \
  --bert_model albert-xxlarge-v2 \
  --output_dir ../../data/checkpoints/20211210_sent_loss20_3e_albert_xxlarge_forwardbest_test_bs24_acc4_truly_5e_6 \
  --model_name AlbertForQuestionAnsweringForwardBest \
  --log_prefix 20211210_sent_loss20_3e_albert_xxlarge_forwardbest_test_bs24_acc4_truly_5e_6 \
  --overwrite_result True \
  --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3_with_entity_label.json \
  --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3_with_entity_label.json \
  --train_supporting_para_file ../../data/hotpot_data/train_golden.json \
  --dev_supporting_para_file ../../data/selector/second_hop_related_paragraph_result/dev_related.json \
  --feature_cache_path ../../data/cache/20211210_sent_loss20_3e_albert_xxlarge_forwardbest_test_bs24_acc4_truly_5e_6 \
  --train_batch_size 24 \
  --gradient_accumulation_steps 4 \
  --local_rank -1 \
  --learning_rate 5e-6 \
  --val_batch_size 128 \
  --save_model_step 1000 \
  --num_train_epochs 3.0
echo "----------------------------------------------------"
