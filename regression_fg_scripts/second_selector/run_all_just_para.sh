echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="1"
# bert-base-uncased
# BertForRelatedSentence
# BertForParagraphClassification
# google/electra-base-discriminator
BERT_MODEL=google/electra-base-discriminator
MODEL_NAME=ElectraForParagraphClassificationTwoRegression
OUTPUT_NAME=20220106_second_fine_grain_epoch10_with_mask
CACHE_NAME=20220106_second_fine_grain_epoch10_with_mask
LOG_PREFIX=20220106_second_fine_grain_epoch10_with_mask
SECOND_PREDICT_PATH=20220106_second_fine_grain_epoch10_with_mask_result
FIRST_PREDICT_PATH=20220105_first_regression_selector_result
CHECKPOINT_PATH=../../data/checkpoints/selector/$OUTPUT_NAME

echo "start train second hop selector..."
cd ../../regression_fg_src/selector
#python -u second_hop_selector.py \
#    --bert_model $BERT_MODEL \
#    --over_write_result True \
#    --output_dir ../../data/checkpoints/selector/$OUTPUT_NAME \
#    --feature_cache_path ../../data/cache/selector/$CACHE_NAME \
#    --log_prefix $LOG_PREFIX \
#    --model_name $MODEL_NAME \
#    --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
#    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
#    --first_predict_result_path ../../data/selector/$FIRST_PREDICT_PATH/ \
#    --best_paragraph_file train_best_paragraph.json \
#    --related_paragraph_file train_top3.json \
#    --new_context_file train_new_context.json \
#    --dev_best_paragraph_file dev_best_paragraph.json \
#    --dev_related_paragraph_file dev_top3.json \
#    --dev_new_context_file dev_new_context.json \
#    --learning_rate 1e-5 \
#    --use_file_cache True \
#    --max_seq_length 512 \
#    --train_batch_size 12 \
#    --val_batch_size 32 \
#    --save_model_step 1000 \
#    --num_train_epochs 10.0
#echo "----------------------------------------------------"
#echo "train second hop selector done!"



cd ../../regression_fg_src/selector
echo "start predict dev second hop result !"
python -u second_hop_selector_predictor.py \
    --bert_model $BERT_MODEL \
    --checkpoint_path $CHECKPOINT_PATH \
    --model_name $MODEL_NAME \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --first_predict_result_path ../../data/selector/$FIRST_PREDICT_PATH \
    --second_predict_result_path ../../data/selector/$SECOND_PREDICT_PATH \
    --final_related_result dev_related.json \
    --best_paragraph_file dev_best_paragraph.json \
    --related_paragraph_file dev_top3.json \
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
    --first_predict_result_path ../../data/selector/$FIRST_PREDICT_PATH \
    --second_predict_result_path ../../data/selector/$SECOND_PREDICT_PATH \
    --final_related_result train_related.json \
    --best_paragraph_file train_best_paragraph.json \
    --related_paragraph_file train_top3.json \
    --new_context_file train_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 16
echo "predict train second hop result done!"

echo "----------------------------------------------------"
echo "train second hop selector done!"
echo "start send email"
cd ../../src/preprocess
python send_email.py $SECOND_PREDICT_PATH predict完成