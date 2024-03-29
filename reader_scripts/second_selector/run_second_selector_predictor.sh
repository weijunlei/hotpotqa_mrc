echo "----------------------------------------------------"
echo "start predict second hop result..."
export CUDA_VISIBLE_DEVICES="2"
cd ../../roberta_src/selector
python -u second_hop_selector_predictor.py \
    --bert_model roberta_large \
     --log_prefix 20211103_roberta_second_hop_selector \
     --log_path ../../log \
    --checkpoint_path ../../data/checkpoints/selector/roberta_second_paragraph_hop_selector \
    --model_name RobertaForRelatedSentence \
    --dev_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --first_predict_result_path ../../data/selector/roberta_first_hop_related_paragraph_result/ \
    --second_predict_result_path ../../data/selector/roberta_second_hop_related_paragraph_result/ \
    --final_related_result train_related.json \
    --best_paragraph_file train_best_paragraph.json \
    --related_paragraph_file train_related_paragraph.json \
    --new_context_file train_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 128
echo "predict train second hop result done!"
echo "start predict dev second hop result !"
python -u second_hop_selector_predictor.py \
    --bert_model roberta-large \
    --log_prefix 20211103_roberta_second_hop_selector \
    --log_path ../../log \
    --checkpoint_path ../../data/checkpoints/selector/roberta_second_paragraph_hop_selector \
    --model_name RobertaForRelatedSentence \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --first_predict_result_path ../../data/selector/roberta_first_hop_related_paragraph_result/ \
    --second_predict_result_path ../../data/selector/roberta_second_hop_related_paragraph_result/ \
    --final_related_result dev_related.json \
    --best_paragraph_file dev_best_paragraph.json \
    --related_paragraph_file dev_related_paragraph.json \
    --new_context_file dev_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 128
echo "predict dev second hop result done!"
echo "----------------------------------------------------"