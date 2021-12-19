echo "----------------------------------------------------"
echo "start predict first hop result..."
export CUDA_VISIBLE_DEVICES="0"
cd ../../src/selector
BERT_MODEL=bert-base-uncased
MODEL_NAME=BertForParagraphClassification
CHECKPOINT_PATH=../../data/checkpoints/selector/20211217_first_hop_bert_simple_paragraph_selector_8_value_setting
FIRST_PREDICT_REUSLt=../../data/selector/20211217_first_hop_bert_simple_paragraph_selector_8_value_setting_result
echo "start predict dev first hop result !"
python -u first_hop_selector_predictor.py \
    --bert_model $BERT_MODEL \
    --checkpoint_path $CHECKPOINT_PATH \
    --model_name $MODEL_NAME \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --predict_result_path $FIRST_PREDICT_REUSLt \
    --best_paragraph_file dev_best_paragraph.json \
    --related_paragraph_file dev_related_paragraph.json \
    --new_context_file dev_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 256
echo "predict dev first hop result done!"
echo "start send email"
cd ../../src/preprocess
python send_email.py $FIRST_PREDICT_REUSLt predict完成
echo "----------------------------------------------------"
echo "start predict train first hop result !"
cd ../../src/selector
python -u first_hop_selector_predictor.py \
    --bert_model $BERT_MODEL \
    --checkpoint_path $CHECKPOINT_PATH \
    --model_name $MODEL_NAME \
    --dev_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --predict_result_path $FIRST_PREDICT_REUSLt \
    --best_paragraph_file train_best_paragraph.json \
    --related_paragraph_file train_related_paragraph.json \
    --new_context_file train_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 256
echo "predict train first hop result done!"
