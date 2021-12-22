echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="2"
echo "start train second hop selector..."
cd ../../roberta_src/selector
python -u second_hop_selector.py \
    --bert_model roberta-large \
    --over_write_result True \
    --log_prefix 20211103_roberta_second_hop_selector \
    --log_path ../../log \
    --output_dir ../../data/checkpoints/selector/roberta_second_paragraph_hop_selector \
    --feature_cache_path ../../data/cache/selector/roberta_second_paragraph_hop_selector \
    --model_name RobertaForRelatedSentence \
    --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --first_predict_result_path ../../data/selector/first_hop_related_paragraph_result/ \
    --best_paragraph_file train_best_paragraph.json \
    --related_paragraph_file train_related_paragraph.json \
    --new_context_file train_new_context.json \
    --dev_best_paragraph_file dev_best_paragraph.json \
    --dev_related_paragraph_file dev_related_paragraph.json \
    --dev_new_context_file dev_new_context.json \
    --use_file_cache True \
    --max_seq_length 512 \
    --train_batch_size 4 \
    --val_batch_size 128 \
    --save_model_step 100 \
    --num_train_epochs 3.0
echo "----------------------------------------------------"
echo "train second hop selector done!"