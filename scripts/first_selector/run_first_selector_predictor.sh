echo "----------------------------------------------------"
echo "start predict second hop result..."
export CUDA_VISIBLE_DEVICES="0"
cd ../../src/selector
echo "start predict train first hop result !"
python -u first_hop_selector_predictor.py \
    --bert_model bert-base-uncased \
    --checkpoint_path ../../data/checkpoints/selector/20211209_first_hop_related_paragraph_selector \
    --model_name BertForRelatedSentence \
    --dev_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --predict_result_path ../../data/selector/20211209_first_hop_related_paragraph_result/ \
    --best_paragraph_file train_best_paragraph.json \
    --related_paragraph_file train_related_paragraph.json \
    --new_context_file train_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 256
echo "predict train first hop result done!"
echo "start predict dev first hop result !"
python -u first_hop_selector_predictor.py \
    --bert_model bert-base-uncased \
    --checkpoint_path ../../data/checkpoints/selector/20211209_first_hop_related_paragraph_selector \
    --model_name BertForRelatedSentence \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --predict_result_path ../../data/selector/20211209_first_hop_related_paragraph_result/ \
    --best_paragraph_file dev_best_paragraph.json \
    --related_paragraph_file dev_related_paragraph.json \
    --new_context_file dev_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 256
echo "predict dev first hop result done!"
echo "----------------------------------------------------"