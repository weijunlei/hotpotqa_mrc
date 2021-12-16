echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="2"
echo "start train first hop selector..."
cd ../../src/selector
# BertForRelatedSentence
# BertForParagraphClassification
python -u first_hop_selector.py \
    --bert_model google/electra-base-discriminator \
    --over_write_result True \
    --output_dir ../../data/checkpoints/selector/20211217_first_hop_electra_related_paragraph_selector_24_value_setting \
    --log_path ../../log \
    --log_prefix 20211217_first_hop_electra_related_paragraph_selector_24_value_setting \
    --feature_cache_path ../../data/cache/selector/20211217_first_hop_electra_related_paragraph_selector_24_value_setting \
    --model_name ElectraForRelatedSentence \
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
python send_email.py 20211214_first_hop_related_paragraph_selector_12 train完成