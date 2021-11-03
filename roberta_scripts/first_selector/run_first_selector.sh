echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="0,1"
echo "start train first hop selector..."
cd ../../roberta_src/selector
python -u first_hop_selector.py \
    --bert_model roberta-large \
    --over_write_result True \
    --use_ddp True \
    --world_size 2 \
    --log_prefix 20211103_first_selector_roberta_large_test \
    --log_path ../../log \
    --output_dir ../../data/checkpoints/selector/20211103_first_selector_roberta_large_test \
    --feature_cache_path ../../data/cache/selector/roberta_first_hop_selector \
    --model_name RobertaForRelatedSentence \
    --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --output_log ../log/roberta_first_hop.txt \
    --use_file_cache True \
    --max_seq_length 512 \
    --train_batch_size 12 \
    --val_batch_size 12 \
    --local_rank 0 \
    --save_model_step 10000 \
    --num_train_epochs 1.0
echo "train first hop selector done!"
echo "----------------------------------------------------"