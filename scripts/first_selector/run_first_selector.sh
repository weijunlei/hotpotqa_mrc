echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="1"
echo "start train first hop selector..."
cd ../../src/selector
# BertForRelatedSentence
# BertForParagraphClassification
python -u first_hop_selector.py \
    --bert_model bert-base-uncased \
    --over_write_result True \
    --output_dir ../../data/checkpoints/selector/20211214_first_hop_just_paragraph_selector_12 \
    --log_path ../../log \
    --log_prefix 20211214_first_hop_just_paragraph_selector_12 \
    --feature_cache_path ../../data/cache/selector/20211214_first_hop_related_paragraph_selector \
    --model_name BertForParagraphClassification \
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