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
               # ElectraForQuestionAnsweringThreeCrossAttention
               # google/electra-large-discriminator
               # ElectraForQuestionAnsweringQANet
cd ../../electra_src/origin_reader
python -u origin_reader_predictor.py \
  --bert_model google/electra-large-discriminator \
  --checkpoint_dir ../../data/checkpoints/20211219_sent_loss_20_2e_electra_large_qa_net_weighted_wo_bi_with_co_cross_with_electra_selector \
  --output_dir ../../data/reader/20211219_test_predict_sent_loss_20_2e_electra_large_qa_net_weighted_wo_bi_with_co_cross_with_electra_selector \
  --predict_file reader_predict.json \
  --model_name ElectraForQuestionAnsweringQANet \
  --log_prefix 20211221_electra_predict_test \
  --overwrite_result True \
  --test_file ../../data/hotpot_data/dev_distractor_input_v1.0.json \
  --test_supporting_para_file ../../data/selector/20211217_second_hop_electra_base_just_paragraph_selector_12_value_setting_result/dev_related.json \
  --feature_cache_path ../../data/cache/20211219_test_sent_loss_20_2e_electra_large_qa_net_weighted_wo_bi_with_co_cross_with_electra_selector \
  --gradient_accumulation_steps 1 \
  --local_rank -1 \
  --learning_rate 2e-5 \
  --val_batch_size 12
echo "----------------------------------------------------"
