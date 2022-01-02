echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="2,0"
echo "start train first hop selector..."
cd ../../combine_src/selector
# ElectraForRelatedSentence
# BertForParagraphClassification
# bert-base-uncased
# google/electra-base-discriminator
BERT_MODEL=google/electra-base-discriminator
OUTPUT_NAME=20211225_first_hop_electra_cross_attention_2e_paragraph_selector_24
LOG_PREFIX=20211225_first_hop_electra_cross_attention_2e_paragraph_selector_24
CACHE_NAME=20211225_first_hop_electra_cross_attention_2e_paragraph_selector_24
MODEL_NAME=ElectraForParagraphClassificationCrossAttention
PREDICT_NAME=20211225_first_hop_electra_cross_attention_2e_paragraph_selector_24_result
python -u first_hop_selector.py \
    --bert_model $BERT_MODEL \
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
    --learning_rate 2e-5 \
    --train_batch_size 24 \
    --val_batch_size 32 \
    --save_model_step 5000 \
    --num_train_epochs 3.0
echo "train first hop selector done!"
echo "----------------------------------------------------"
echo "start send email"
cd ../../combine_src/preprocess
python send_email.py $OUTPUT_NAME train完成

echo "----------------------------------------------------"
echo "start predict first hop result..."
echo "start predict dev first hop result !"
cd ../../combine_src/selector
python -u first_hop_selector_predictor.py \
    --bert_model $BERT_MODEL \
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
cd ../../combine_src/preprocess
python send_email.py 20211213_first_hop_related_paragraph_selector predict完成
echo "----------------------------------------------------"
echo "start predict train first hop result !"
cd ../../combine_src/selector
python -u first_hop_selector_predictor.py \
    --bert_model $BERT_MODEL \
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
