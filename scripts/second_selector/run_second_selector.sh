echo "----------------------------------------------------"
MODEL_NAME=
export CUDA_VISIBLE_DEVICES="0"
echo "start train first hop selector..."
cd ../../src/selector
python -u second_hop_selector.py \
    --bert_model bert-base-uncased \
    --over_write_result True \
    --output_dir ../../data/checkpoints/selector/20211217_second_related_paragraph_hop_selector_truly_value_setting \
    --feature_cache_path ../../data/cache/selector/20211217_second_related_paragraph_hop_selector_truly_value_setting \
    --log_prefix 20211217_second_related_paragraph_hop_selector_truly_value_setting \
    --model_name BertForRelatedSentence \
    --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --first_predict_result_path ../../data/selector/20211213_first_hop_related_paragraph_result/ \
    --best_paragraph_file train_best_paragraph.json \
    --related_paragraph_file train_related_paragraph.json \
    --new_context_file train_new_context.json \
    --dev_best_paragraph_file dev_best_paragraph.json \
    --dev_related_paragraph_file dev_related_paragraph.json \
    --dev_new_context_file dev_new_context.json \
    --use_file_cache True \
    --max_seq_length 512 \
    --train_batch_size 12 \
    --val_batch_size 128 \
    --save_model_step 10000 \
    --num_train_epochs 3.0
echo "----------------------------------------------------"
echo "train second hop selector done!"
echo "start send email"
cd ../../src/preprocess
python send_email.py 20211214_second_related_paragraph_hop_selector_truly train完成