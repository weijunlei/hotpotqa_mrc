echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="0"
echo "start train first hop selector..."
cd ../../roberta_src/selector
python -u first_hop_selector.py \
    --bert_model roberta-large \
    --over_write_result True \
    --output_dir ../../data/checkpoints/selector/roberta_first_hop_selector \
    --feature_cache_path ../../data/cache/selector/roberta_first_hop_selector \
    --model_name RobertaForRelatedSentence \
    --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --output_log ../log/roberta_first_hop.txt \
    --use_file_cache True \
    --max_seq_length 512 \
    --train_batch_size 24 \
    --val_batch_size 128 \
    --save_model_step 50000 \
    --num_train_epochs 3.0
echo "train first hop selector done!"
echo "----------------------------------------------------"