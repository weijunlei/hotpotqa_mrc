echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="2,3"
echo "start train first hop selector..."
cd ../../electra_src/selector
python -u first_hop_selector.py \
    --bert_model google/electra-base-discriminator \
    --over_write_result True \
    --output_dir ../../data/checkpoints/selector/20211120_electra_base \
    --feature_cache_path ../../data/cache/selector/20211120_electra_base \
    --model_name ElectraForParagraphClassification \
    --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --output_log ../log/20211120_electra_base.txt \
    --use_file_cache True \
    --max_seq_length 512 \
    --train_batch_size 12 \
    --val_batch_size 24 \
    --save_model_step 1000 \
    --learning_rate 1e-5 \
    --num_train_epochs 3.0
echo "train first hop selector done!"
echo "----------------------------------------------------"