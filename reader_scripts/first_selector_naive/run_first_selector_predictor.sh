echo "----------------------------------------------------"
echo "start predict first hop result..."
export CUDA_VISIBLE_DEVICES="1"
cd ../../roberta_src/naive_selector
echo "start predict dev first hop result !"
#python -u first_select_predictor.py \
#    --bert_model roberta-large \
#    --checkpoint_path ../../data/checkpoints/selector/20211105_roberta_large_first_hop_just_paragraph_selector_test \
#    --feature_cache_path ../../data/cache/20211106_first_hop_selector/ \
#    --log_prefix 20211106_first_selector_predict_roberta_large_test \
#    --log_path ../../log \
#    --model_name RobertaForParagraphClassification \
#    --predict_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
#    --predict_result_path ../../data/selector/roberta_first_hop_related_paragraph_result/ \
#    --best_paragraph_file dev_best_paragraph.json \
#    --all_paragraph_file dev_all_paragraph.json \
#    --max_seq_length 512 \
#    --val_batch_size 120
echo "predict dev first hop result done!"
echo "----------------------------------------------------"
#echo "start predict train first hop result !"
python -u first_select_predictor.py \
    --bert_model roberta-large \
    --checkpoint_path ../../data/checkpoints/selector/20211105_roberta_large_first_hop_just_paragraph_selector_test \
    --feature_cache_path ../../data/cache/20211106_first_hop_selector/ \
    --log_prefix 20211106_first_selector_predict_roberta_large_test \
    --log_path ../../log \
    --model_name RobertaForParagraphClassification \
    --predict_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --predict_result_path ../../data/selector/roberta_first_hop_related_paragraph_result/ \
    --best_paragraph_file train_best_paragraph.json \
    --all_paragraph_file train_all_paragraph.json \
    --max_seq_length 512 \
    --val_batch_size 120
#echo "predict train first hop result done!"
