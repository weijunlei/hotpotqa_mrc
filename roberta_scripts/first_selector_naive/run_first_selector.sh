echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="0,1,2"
echo "start train first hop selector..."
cd ../../roberta_src/naive_selector
python -u first_select_trainer.py \
    --bert_model roberta-large \
    --use_ddp False \
    --world_size 1 \
    --local_rank -1 \
    --feature_cache_path ../../data/cache/selector/20211105_roberta_large_naive \
    --log_prefix 20211105_roberta_large_naive \
    --log_path ../../log \
    --output_dir ../../data/checkpoints/selector/20211105_roberta_large_first_hop_just_paragraph_selector_test \
    --model_name RobertaForParagraphClassification \
    --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --doc_stride 256 \
    --max_seq_length 512 \
    --train_batch_size 36 \
    --val_batch_size 48 \
    --save_model_step 50 \
    --num_train_epochs 3.0
echo "train first hop selector done!"
echo "----------------------------------------------------"