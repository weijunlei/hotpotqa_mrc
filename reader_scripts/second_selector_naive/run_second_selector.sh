echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="0,1,2"
echo "start train second hop selector..."
cd ../../roberta_src/naive_second_selector
python -u second_select_trainer.py \
    --bert_model roberta-base \
    --use_ddp True \
    --world_size 3 \
    --local_rank 0 \
    --feature_cache_path ../../data/cache/selector/20211106_second_roberta_base_naive \
    --log_prefix 20211106_second_roberta_base_naive \
    --log_path ../../log \
    --output_dir ../../data/checkpoints/selector/20211106_roberta_base_second_hop_just_paragraph_selector_test \
    --first_predict_result_path ../../data/selector/first_hop_just_paragraph_result \
    --best_train_paragraph_file train_best_paragraph.json \
    --best_dev_paragraph_file dev_best_paragraph.json \
    --model_name RobertaForParagraphClassification \
    --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --doc_stride 256 \
    --max_seq_length 512 \
    --train_batch_size 4 \
    --val_batch_size 48 \
    --save_model_step 10 \
    --num_train_epochs 3.0
echo "train first hop selector done!"
echo "----------------------------------------------------"