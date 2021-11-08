echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="0,1,2"
echo "start train first hop selector..."
cd ../../roberta_src/naive_selector
#python -u first_select_trainer.py \
#    --bert_model roberta-base \
#    --use_ddp True \
#    --world_size 3 \
#    --local_rank 0 \
#    --feature_cache_path ../../data/cache/selector/20211107_roberta_base_naive \
#    --log_prefix 20211107_roberta_base_naive \
#    --log_path ../../log \
#    --output_dir ../../data/checkpoints/selector/20211107_roberta_base_first_hop_just_paragraph_selector_naive \
#    --model_name RobertaForParagraphClassification \
#    --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
#    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
#    --doc_stride 256 \
#    --max_seq_length 512 \
#    --train_batch_size 12 \
#    --val_batch_size 48 \
#    --save_model_step 5000 \
#    --num_train_epochs 3.0
#echo "train first hop selector done!"
#echo "----------------------------------------------------"

export CUDA_VISIBLE_DEVICES="0"
echo "start predict dev first hop result !"
python -u first_select_predictor.py \
    --bert_model roberta-base \
    --checkpoint_path ../../data/checkpoints/selector/20211107_roberta_base_first_hop_just_paragraph_selector_naive \
    --feature_cache_path ../../data/cache/20211107_first_hop_selector/ \
    --log_prefix 20211107_first_hop_selector \
    --log_path ../../log \
    --model_name RobertaForParagraphClassification \
    --predict_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --predict_result_path ../../data/selector/20211107_roberta_first_hop_related_paragraph_result/ \
    --best_paragraph_file dev_best_paragraph.json \
    --all_paragraph_file dev_all_paragraph.json \
    --max_seq_length 512 \
    --val_batch_size 48
echo "predict dev first hop result done!"
echo "----------------------------------------------------"
echo "start predict train first hop result !"
python -u first_select_predictor.py \
    --bert_model roberta-base \
    --checkpoint_path ../../data/checkpoints/selector/20211107_roberta_base_first_hop_just_paragraph_selector_naive \
    --feature_cache_path ../../data/cache/20211107_first_hop_selector/ \
    --log_prefix 20211107_first_hop_selector \
    --log_path ../../log \
    --model_name RobertaForParagraphClassification \
    --predict_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --predict_result_path ../../data/selector/20211107_roberta_first_hop_related_paragraph_result/ \
    --best_paragraph_file train_best_paragraph.json \
    --all_paragraph_file train_all_paragraph.json \
    --max_seq_length 512 \
    --val_batch_size 48
echo "predict train first hop result done!"


echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="0"
cd ../../roberta_src/naive_second_selector
echo "start train second hop selector..."
python -u second_select_trainer.py \
    --bert_model roberta-base \
    --use_ddp False \
    --world_size 3 \
    --local_rank -1 \
    --feature_cache_path ../../data/cache/selector/20211107_second_roberta_base_naive \
    --log_prefix 20211107_second_roberta_base_naive \
    --log_path ../../log \
    --output_dir ../../data/checkpoints/selector/20211107_roberta_base_second_hop_just_paragraph_selector_test \
    --first_predict_result_path ../../data/selector/20211107_roberta_first_hop_related_paragraph_result \
    --best_train_paragraph_file train_best_paragraph.json \
    --best_dev_paragraph_file dev_best_paragraph.json \
    --model_name RobertaForParagraphClassification \
    --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --doc_stride 256 \
    --max_seq_length 512 \
    --train_batch_size 12 \
    --val_batch_size 48 \
    --save_model_step 5000 \
    --num_train_epochs 3.0
echo "train first hop selector done!"
echo "----------------------------------------------------"