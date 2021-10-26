echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="0,1"
echo "start train first hop selector..."
cd ../../src/selector
python -u second_hop_selector.py \
    --bert_model roberta-large \
    --over_write_result True \
    --output_dir ../../data/checkpoints/selector/roberta_second_paragraph_hop_selector \
    --feature_cache_path ../../data/cache/selector/roberta_second_paragraph_hop_selector \
    --model_name RobertaForRelatedSentence \
    --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --first_predict_result_path ../../data/selector/roberta_first_hop_related_paragraph_result/ \
    --best_paragraph_file train_best_paragraph.json \
    --related_paragraph_file train_related_paragraph.json \
    --new_context_file train_new_context.json \
    --dev_best_paragraph_file dev_best_paragraph.json \
    --dev_related_paragraph_file dev_related_paragraph.json \
    --dev_new_context_file dev_new_context.json \
    --output_log ../log/selector_two_hop.txt \
    --use_file_cache True \
    --max_seq_length 512 \
    --train_batch_size 12 \
    --val_batch_size 128 \
    --save_model_step 50000 \
    --num_train_epochs 3.0
echo "----------------------------------------------------"
echo "train second hop selector done!"