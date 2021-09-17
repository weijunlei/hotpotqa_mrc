echo "----------------------------------------------------"
echo "start predict second hop result..."
export CUDA_VISIBLE_DEVICES="2,3"
cd ../selector
echo "start predict train first hop result !"
python -u first_hop_selector_predictor.py \
    --bert_model bert-base-uncased \
    --checkpoint_path ../checkpoints/selector/first_hop_selector \
    --model_name BertForRelated \
    --dev_file ../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --predict_result_path ../data/selector/first_hop_result/ \
    --best_paragraph_file train_best_paragraph.json \
    --related_paragraph_file train_related_paragraph.json \
    --new_context_file train_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 128
echo "predict train first hop result done!"
echo "start predict dev first hop result !"
python -u first_hop_selector_predictor.py \
    --bert_model bert-base-uncased \
    --checkpoint_path ../checkpoints/selector/first_hop_selector \
    --model_name BertForRelated \
    --dev_file ../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --predict_result_path ../data/selector/first_hop_result/ \
    --best_paragraph_file dev_best_paragraph.json \
    --related_paragraph_file dev_related_paragraph.json \
    --new_context_file dev_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 128
echo "predict dev first hop result done!"
echo "----------------------------------------------------"