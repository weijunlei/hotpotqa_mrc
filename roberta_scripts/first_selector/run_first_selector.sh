echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="0,1,2"
echo "start train first hop selector..."
cd ../../roberta_src/selector
python -u first_hop_selector.py \
    --bert_model roberta-large \
    --output_dir ../../data/checkpoints/selector/20211105_roberta_base_first_hop_just_paragraph_selector_test \
    --model_name RobertaForRelatedSentence \
    --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --output_log ../../log/20211105_roberta_base_selector_1_paragraph_base_2e-5.txt \
    --max_seq_length 512 \
    --train_batch_size 48 \
    --val_batch_size 36 \
    --save_model_step 500 \
    --num_train_epochs 3.0
echo "train first hop selector done!"
echo "----------------------------------------------------"