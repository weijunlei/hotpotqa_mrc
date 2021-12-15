echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="2"
echo "start train first hop selector..."
cd ../../src/selector
# BertForRelatedSentence
# BertForParagraphClassification
OUTPUT_NAME=20211214_first_hop_just_paragraph_selector_12
LOG_PREFIX=20211214_first_hop_just_paragraph_selector_12
CACHE_NAME=20211214_first_hop_just_paragraph_selector_12
MODEL_NAME=BertForParagraphClassification
PREDICT_NAME=20211214_first_hop_just_paragraph_result
python -u first_hop_selector.py \
    --bert_model bert-base-uncased \
    --over_write_result True \
    --output_dir ../../data/checkpoints/selector/$OUTPUT_NAME \
    --log_path ../../log \
    --log_prefix $LOG_PREFIX \
    --feature_cache_path ../../data/cache/selector/$CACHE_NAME \
    --model_name $MODEL_NAME \
    --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --use_file_cache True \
    --max_seq_length 512 \
    --train_batch_size 12 \
    --val_batch_size 64 \
    --save_model_step 10000 \
    --num_train_epochs 3.0
echo "train first hop selector done!"
echo "----------------------------------------------------"
echo "start send email"
cd ../../src/preprocess
python send_email.py $OUTPUT_NAME train完成

echo "----------------------------------------------------"
echo "start predict first hop result..."
cd ../../src/selector
echo "start predict dev first hop result !"
cd ../../src/selector
python -u first_hop_selector_predictor.py \
    --bert_model bert-base-uncased \
    --checkpoint_path ../../data/checkpoints/selector/$OUTPUT_NAME \
    --model_name $MODEL_NAME \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --predict_result_path ../../data/selector/$PREDICT_NAME/ \
    --best_paragraph_file dev_best_paragraph.json \
    --related_paragraph_file dev_related_paragraph.json \
    --new_context_file dev_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 256
echo "predict dev first hop result done!"
echo "start send email"
cd ../../src/preprocess
python send_email.py 20211213_first_hop_related_paragraph_selector predict完成
echo "----------------------------------------------------"
echo "start predict train first hop result !"
cd ../../src/selector
python -u first_hop_selector_predictor.py \
    --bert_model bert-base-uncased \
    --checkpoint_path ../../data/checkpoints/selector/$OUTPUT_NAME \
    --model_name $MODEL_NAME \
    --dev_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --predict_result_path ../../data/selector/$PREDICT_NAME/ \
    --best_paragraph_file train_best_paragraph.json \
    --related_paragraph_file train_related_paragraph.json \
    --new_context_file train_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 256
echo "predict train first hop result done!"
