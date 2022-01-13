echo "----------------------------------------------------"
# ElectraForRelatedSentence
# BertForParagraphClassification
# bert-base-uncased
# google/electra-base-discriminator
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME=ElectraForParagraphClassification
BERT_MODEL=google/electra-base-discriminator
CHECKPOINT_PATH=../../data/checkpoints/selector/20211217_first_hop_electra_base_just_paragraph_selector_12_value_setting
PREDICT_RESULT_PATH=../../data/selector/20220104
echo "start predict first hop result..."
export CUDA_VISIBLE_DEVICES="0"
cd ../../src/selector
echo "start predict dev first hop result !"
python -u first_hop_selector_predictor.py \
    --bert_model $BERT_MODEL \
    --checkpoint_path $CHECKPOINT_PATH \
    --model_name $MODEL_NAME \
    --dev_file ../../data/hotpot_data/hotpot_dev_distractor_v1.json \
    --predict_result_path $PREDICT_RESULT_PATH \
    --best_paragraph_file dev_best_paragraph.json \
    --related_paragraph_file dev_related_paragraph.json \
    --new_context_file dev_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 32
