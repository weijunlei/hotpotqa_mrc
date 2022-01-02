echo "----------------------------------------------------"
echo "start predict second hop result..."
export CUDA_VISIBLE_DEVICES="1"
# bert-base-uncased
# BertForRelatedSentence
# google/electra-base-discriminator
MODEL_NAME=ElectraForParagraphClassification
BERT_MODEL=google/electra-base-discriminator
CHECKPOINT_PATH=../../data/checkpoints/selector/20211217_second_hop_electra_base_just_paragraph_selector_12_value_setting
FIRST_PREDICT_PATH=../../data/selector/20211217_first_hop_electra_base_just_paragraph_selector_12_value_setting_result/
#OUTPUT_NAME=20211217_second_hop_electra_base_related_paragraph_selector_12_value_setting
#CACHE_NAME=20211217_second_hop_electra_base_related_paragraph_selector_12_value_setting
#LOG_PREFIX=20211217_second_hop_electra_base_related_paragraph_selector_12_value_setting
SECOND_PREDICT_PATH=../../data/selector/20211217_second_hop_electra_base_just_paragraph_selector_12_value_setting_test_result
cd ../../src/selector
echo "start predict dev second hop result !"
python -u second_hop_selector_predictor.py \
    --bert_model $BERT_MODEL \
    --checkpoint_path $CHECKPOINT_PATH \
    --model_name $MODEL_NAME \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --first_predict_result_path $FIRST_PREDICT_PATH \
    --second_predict_result_path $SECOND_PREDICT_PATH \
    --final_related_result dev_related.json \
    --best_paragraph_file dev_best_paragraph.json \
    --related_paragraph_file dev_related_paragraph.json \
    --new_context_file dev_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 16
echo "predict dev second hop result done!"
echo "----------------------------------------------------"
python -u second_hop_selector_predictor.py \
    --bert_model $BERT_MODEL \
    --checkpoint_path $CHECKPOINT_PATH \
    --model_name $MODEL_NAME \
    --dev_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --first_predict_result_path $FIRST_PREDICT_PATH \
    --second_predict_result_path $SECOND_PREDICT_PATH \
    --final_related_result train_related.json \
    --best_paragraph_file train_best_paragraph.json \
    --related_paragraph_file train_related_paragraph.json \
    --new_context_file train_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 16
echo "predict train second hop result done!"

echo "----------------------------------------------------"
echo "train second hop selector done!"
echo "start send email"
cd ../../src/preprocess
python send_email.py $SECOND_PREDICT_PATH predict完成
