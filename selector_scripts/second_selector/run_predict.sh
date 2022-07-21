echo "----------------------------------------------------"
echo "start predict second hop result..."
export CUDA_VISIBLE_DEVICES="0"
# bert-base-uncased
# BertForRelatedSentence
# google/electra-base-discriminator
MODEL_NAME=ElectraForParagraphClassification
BERT_MODEL=google/electra-base-discriminator
CHECKPOINT_PATH=../../data/checkpoints/selector/754/second_hop_model
FIRST_PREDICT_PATH=../../data/result/754/first_selector/
SECOND_PREDICT_PATH=../../data/result/754/second_selector/
cd ../../src/selector
echo "start predict dev second hop result !"
python -u second_hop_selector_predictor.py \
    --bert_model $BERT_MODEL \
    --checkpoint_path $CHECKPOINT_PATH \
    --model_name $MODEL_NAME \
    --dev_file ../../data/hotpot_data/hotpot_dev_distractor_v1.json \
    --first_predict_result_path $FIRST_PREDICT_PATH \
    --second_predict_result_path $SECOND_PREDICT_PATH \
    --final_related_result dev_related.json \
    --best_paragraph_file dev_best_paragraph.json \
    --related_paragraph_file dev_related_paragraph.json \
    --new_context_file dev_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 16
echo "predict dev second hop result done!"
