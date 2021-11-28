echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="2"
echo "start train first hop selector..."
cd ../../electra_src/electra_classification
python -u first_hop_selector.py \
    --bert_model roberta-base \
    --overwrite_result True \
    --log_prefix 20211128_electra_base_sequence_classification_sample_1 \
    --log_path ../../log \
    --output_dir ../../data/checkpoints/selector/20211128_electra_base_sequence_classification_sample_1 \
    --feature_cache_path ../../data/cache/selector/20211128_electra_base_sequence_classification_sample_1 \
    --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --summary ../../log/20211125_electra_small_sequence_classification \
    --evaluate_during_training True \
    --per_gpu_train_batch_size 8 \
    --use_file_cache True \
    --max_seq_length 512 \
    --val_batch_size 48 \
    --learning_rate 1e-5 \
    --logging_steps 10000 \
    --save_model_step 10000 \
    --num_train_epochs 2.0
echo "train first hop selector done!"
echo "----------------------------------------------------"