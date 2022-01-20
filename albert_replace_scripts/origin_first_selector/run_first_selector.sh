echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="2"
echo "start train first hop selector..."
cd ../../electra_src/bert_selector
python -u first_hop_selector.py \
    --bert_model bert-base-uncased \
    --over_write_result True \
    --output_dir ../../data/checkpoints/selector/20211123_first_hop_just_paragraph_selector_test \
    --feature_cache_path ../../data/cache/selector/20211123_first_hop_just_paragraph_selector_test \
    --model_name BertForParagraphClassification \
    --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --output_log ../log/selector_1_paragraph_base_2e-5.txt \
    --use_file_cache True \
    --max_seq_length 512 \
    --train_batch_size 4 \
    --val_batch_size 4 \
    --save_model_step 1000 \
    --num_train_epochs 3.0
echo "train first hop selector done!"
echo "----------------------------------------------------"