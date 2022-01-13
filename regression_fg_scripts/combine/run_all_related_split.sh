echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="1,2"
echo "start train first hop selector..."
cd ../../src/selector
# BertForRelatedSentence
# BertForParagraphClassification
OUTPUT_NAME=20211214_first_hop_related_paragraph_selector_12_split
LOG_PREFIX=20211214_first_hop_related_paragraph_selector_12_split
CACHE_NAME=20211214_first_hop_related_paragraph_selector_12_split
MODEL_NAME=BertForRelatedSentence
PREDICT_NAME=20211214_first_hop_related_paragraph_result_split
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

export CUDA_VISIBLE_DEVICES="1"
echo "----------------------------------------------------"
echo "start predict first hop result..."
cd ../../src/selector
echo "start predict dev first hop result !"
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
cd ../../src/selector
echo "start predict train first hop result !"
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

echo "start send email"
cd ../../src/preprocess
python send_email.py $OUTPUT_NAME predict完成


export CUDA_VISIBLE_DEVICES="1,2"
echo "----------------------------------------------------"
# BertForRelatedSentence
# BertForParagraphClassification
MODEL_NAME=BertForRelatedSentence
OUTPUT_NAME=20211214_second_related_paragraph_hop_selector_truly_split
CACHE_NAME=20211214_second_related_paragraph_hop_selector_truly_split
LOG_PREFIX=20211214_second_related_paragraph_hop_selector_truly_split
FIRST_PREDICT_PATH=20211214_first_hop_related_paragraph_result_split
SECOND_PREDICT_PATH=20211214_second_related_paragraph_hop_selector_truly_split

echo "start train second hop selector..."
cd ../../src/selector
python -u second_hop_selector.py \
    --bert_model bert-base-uncased \
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
    --val_batch_size 128 \
    --save_model_step 10000 \
    --num_train_epochs 3.0
echo "----------------------------------------------------"
echo "train second hop selector done!"
echo "start send email"
cd ../../src/preprocess
python send_email.py $OUTPUT_NAME train完成
export CUDA_VISIBLE_DEVICES="1"
echo "----------------------------------------------------"
echo "start predict second hop result..."
cd ../../src/selector
echo "start predict dev second hop result !"
python -u second_hop_selector_predictor.py \
    --bert_model bert-base-uncased \
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
    --bert_model bert-base-uncased \
    --checkpoint_path ../../data/checkpoints/selector/$OUTPUT_NAME \
    --model_name BertForRelatedSentence \
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
cd ../../src/preprocess
python send_email.py $OUTPUT_NAME predict完成
