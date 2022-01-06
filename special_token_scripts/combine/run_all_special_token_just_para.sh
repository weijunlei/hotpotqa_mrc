echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="2"
echo "start train first hop selector..."
cd ../../special_token_src/selector
# ElectraForRelatedSentence
# BertForParagraphClassification
# bert-base-uncased
# google/electra-base-discriminator
BERT_MODEL=google/electra-base-discriminator
#OUTPUT_NAME=20220105_special_token_cls_first_selector
#LOG_PREFIX=20220105_special_token_cls_first_selector
#CACHE_NAME=20220105_special_token_cls_first_selector
MODEL_NAME=ElectraForParagraphClassification
#PREDICT_NAME=20220105_special_token_cls_first_selector_result
#python -u first_hop_selector.py \
#    --bert_model $BERT_MODEL \
#    --over_write_result True \
#    --output_dir ../../data/checkpoints/selector/$OUTPUT_NAME \
#    --log_path ../../log \
#    --log_prefix $LOG_PREFIX \
#    --feature_cache_path ../../data/cache/selector/$CACHE_NAME \
#    --model_name $MODEL_NAME \
#    --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
#    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
#    --use_file_cache True \
#    --max_seq_length 512 \
#    --learning_rate 1e-5 \
#    --train_batch_size 12 \
#    --val_batch_size 16 \
#    --save_model_step 10000 \
#    --num_train_epochs 3.0
#echo "train first hop selector done!"
#echo "----------------------------------------------------"
#echo "start send email"
#cd ../../special_token_src/preprocess
#python send_email.py $OUTPUT_NAME train完成
#
#echo "----------------------------------------------------"
#echo "start predict first hop result..."
#cd ../../special_token_src/selector
#echo "start predict dev first hop result !"
#cd ../../special_token_src/selector
#python -u first_hop_selector_predictor.py \
#    --bert_model $BERT_MODEL \
#    --checkpoint_path ../../data/checkpoints/selector/$OUTPUT_NAME \
#    --model_name $MODEL_NAME \
#    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
#    --predict_result_path ../../data/selector/$PREDICT_NAME/ \
#    --best_paragraph_file dev_best_paragraph.json \
#    --related_paragraph_file dev_related_paragraph.json \
#    --new_context_file dev_new_context.json \
#    --max_seq_length 512 \
#    --val_batch_size 32
#echo "predict dev first hop result done!"
#echo "start send email"
#cd ../../special_token_src/preprocess
#python send_email.py 20211213_first_hop_related_paragraph_selector predict完成
#echo "----------------------------------------------------"
#echo "start predict train first hop result !"
#cd ../../special_token_src/selector
#python -u first_hop_selector_predictor.py \
#    --bert_model $BERT_MODEL \
#    --checkpoint_path ../../data/checkpoints/selector/$OUTPUT_NAME \
#    --model_name $MODEL_NAME \
#    --dev_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
#    --predict_result_path ../../data/selector/$PREDICT_NAME/ \
#    --best_paragraph_file train_best_paragraph.json \
#    --related_paragraph_file train_related_paragraph.json \
#    --new_context_file train_new_context.json \
#    --max_seq_length 512 \
#    --val_batch_size 32
#echo "predict train first hop result done!"




echo "----------------------------------------------------"
# bert-base-uncased
# BertForRelatedSentence
# BertForParagraphClassification
# google/electra-base-discriminator
OUTPUT_NAME=20220105_special_token_cls_second_selector
CACHE_NAME=20220105_special_token_cls_second_selector
LOG_PREFIX=20220105_special_token_cls_second_selector
FIRST_PREDICT_PATH=20220105_special_token_cls_first_selector_result
SECOND_PREDICT_PATH=20220105_special_token_cls_second_selector_result

echo "start train second hop selector..."
cd ../../special_token_src/selector
python -u second_hop_selector.py \
    --bert_model $BERT_MODEL \
    --over_write_result True \
    --output_dir ../../data/checkpoints/selector/$OUTPUT_NAME \
    --feature_cache_path ../../data/cache/selector/$CACHE_NAME \
    --log_prefix $LOG_PREFIX \
    --model_name $MODEL_NAME \
    --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --first_predict_result_path ../../data/selector/$FIRST_PREDICT_PATH/ \
    --best_paragraph_file train_best_paragraph.json \
    --related_paragraph_file train_related_paragraph.json \
    --new_context_file train_new_context.json \
    --dev_best_paragraph_file dev_best_paragraph.json \
    --dev_related_paragraph_file dev_related_paragraph.json \
    --dev_new_context_file dev_new_context.json \
    --use_file_cache True \
    --max_seq_length 512 \
    --train_batch_size 12 \
    --val_batch_size 16 \
    --save_model_step 10000 \
    --num_train_epochs 3.0
echo "----------------------------------------------------"
echo "train second hop selector done!"
echo "start send email"
cd ../../special_token_src/preprocess
python send_email.py $OUTPUT_NAME train完成

echo "----------------------------------------------------"
echo "start predict second hop result..."
cd ../../special_token_src/selector
echo "start predict dev second hop result !"
python -u second_hop_selector_predictor.py \
    --bert_model $BERT_MODEL \
    --checkpoint_path ../../data/checkpoints/selector/$OUTPUT_NAME \
    --model_name $MODEL_NAME \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --first_predict_result_path ../../data/selector/$FIRST_PREDICT_PATH/ \
    --second_predict_result_path ../../data/selector/$SECOND_PREDICT_PATH/ \
    --final_related_result dev_related.json \
    --best_paragraph_file dev_best_paragraph.json \
    --related_paragraph_file dev_related_paragraph.json \
    --new_context_file dev_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 128
echo "predict dev second hop result done!"
echo "----------------------------------------------------"
python -u second_hop_selector_predictor.py \
    --bert_model $BERT_MODEL \
    --checkpoint_path ../../data/checkpoints/selector/$OUTPUT_NAME \
    --model_name $MODEL_NAME \
    --dev_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --first_predict_result_path ../../data/selector/$FIRST_PREDICT_PATH/ \
    --second_predict_result_path ../../data/selector/$SECOND_PREDICT_PATH/ \
    --final_related_result train_related.json \
    --best_paragraph_file train_best_paragraph.json \
    --related_paragraph_file train_related_paragraph.json \
    --new_context_file train_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 128
echo "predict train second hop result done!"

echo "----------------------------------------------------"
echo "train second hop selector done!"
echo "start send email"
cd ../../special_token_src/preprocess
python send_email.py $OUTPUT_NAME predict完成