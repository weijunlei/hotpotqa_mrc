echo "----------------------------------------------------"
# ElectraForRelatedSentence
# BertForParagraphClassification
# bert-base-uncased
# google/electra-base-discriminator
MODEL_NAME=BertForParagraphClassification
BERT_MODEL=bert-base-uncased
CHECKPOINT_PATH=../../data/checkpoints/selector/20211214_second_just_paragraph_hop_selector_truly_debug
PREDICT_RESULT_PATH=../../data/selector/20211214_second_just_paragraph_hop_selector_truly_debug_result
echo "start predict first hop result..."
export CUDA_VISIBLE_DEVICES="0"
cd ../../src/selector
echo "start predict dev first hop result !"
python -u first_hop_selector_predictor.py \
    --bert_model $BERT_MODEL \
    --checkpoint_path $CHECKPOINT_PATH \
    --model_name $MODEL_NAME \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --predict_result_path $PREDICT_RESULT_PATH \
    --best_paragraph_file dev_best_paragraph.json \
    --related_paragraph_file dev_related_paragraph.json \
    --new_context_file dev_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 32
echo "predict dev first hop result done!"
echo "start send email"
cd ../../src/preprocess
python send_email.py $CHECKPOINT_PATH完成
echo "----------------------------------------------------"
echo "start predict train first hop result !"
cd ../../src/selector
python -u first_hop_selector_predictor.py \
    --bert_model $BERT_MODEL \
    --checkpoint_path $CHECKPOINT_PATH \
    --model_name $MODEL_NAME \
    --dev_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --predict_result_path $PREDICT_RESULT_PATH \
    --best_paragraph_file train_best_paragraph.json \
    --related_paragraph_file train_related_paragraph.json \
    --new_context_file train_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 32
echo "predict train first hop result done!"
