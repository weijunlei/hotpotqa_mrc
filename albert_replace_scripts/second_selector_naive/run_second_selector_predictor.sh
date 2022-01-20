echo "----------------------------------------------------"
echo "start predict first hop result..."
export CUDA_VISIBLE_DEVICES="0"
#cd ../../roberta_src/naive_second_selector
#echo "start predict dev first hop result !"
#python -u second_select_predictor.py \
#    --bert_model roberta-base \
#    --checkpoint_path ../../data/checkpoints/selector/20211107_roberta_base_first_hop_just_paragraph_selector_naive \
#    --feature_cache_path ../../data/cache/20211107_second_hop_selector/ \
#    --log_prefix 20211107_roberta_base_second_hop_just_paragraph_selector_test \
#    --log_path ../../log \
#    --model_name RobertaForParagraphClassification \
#    --predict_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
#    --predict_result_path ../../data/selector/roberta_second_hop_related_paragraph_result/ \
#    --first_predict_result_path ../../data/selector/20211107_roberta_first_hop_related_paragraph_result \
#    --best_paragraph_file dev_best_paragraph.json \
#    --best_relate_paragraph_file dev_related.json \
#    --all_paragraph_file dev_all_paragraph.json \
#    --max_seq_length 512 \
#    --val_batch_size 120
#echo "predict dev first hop result done!"\



cd ../../roberta_src/naive_second_selector
echo "start predict dev first hop result !"
python -u second_select_predictor.py \
    --bert_model roberta-base \
    --checkpoint_path ../../data/checkpoints/selector/20211107_roberta_base_first_hop_just_paragraph_selector_naive \
    --feature_cache_path ../../data/cache/20211107_second_hop_selector/ \
    --log_prefix 20211107_roberta_base_second_hop_just_paragraph_selector_test \
    --log_path ../../log \
    --model_name RobertaForParagraphClassification \
    --predict_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --predict_result_path ../../data/selector/roberta_second_hop_related_paragraph_result/ \
    --first_predict_result_path ../../data/selector/20211107_roberta_first_hop_related_paragraph_result \
    --best_paragraph_file train_best_paragraph.json \
    --best_relate_paragraph_file train_related.json \
    --all_paragraph_file train_all_paragraph.json \
    --max_seq_length 512 \
    --val_batch_size 120
echo "predict dev first hop result done!"
