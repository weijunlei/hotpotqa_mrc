echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="1"
echo "start train first hop selector..."
cd ../../roberta_src/selector
python -u first_hop_selector.py \
    --bert_model roberta-large \
    --over_write_result True \
    --output_dir ../../data/checkpoints/selector/roberta_first_hop_selector \
    --feature_cache_path ../../data/cache/selector/roberta_first_hop_selector \
    --model_name RobertaForRelatedSentence \
    --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --output_log ../log/roberta_first_hop.txt \
    --use_file_cache True \
    --max_seq_length 512 \
    --train_batch_size 12 \
    --val_batch_size 128 \
    --save_model_step 10000 \
    --num_train_epochs 3.0
echo "train first hop selector done!"
echo "----------------------------------------------------"


echo "----------------------------------------------------"
echo "start predict second hop result..."
echo "start predict train first hop result !"
python -u first_hop_selector_predictor.py \
    --bert_model roberta-large \
    --checkpoint_path ../../data/checkpoints/selector/roberta_first_hop_selector \
    --model_name RobertaForRelatedSentence \
    --dev_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --predict_result_path ../../data/selector/roberta_first_hop_related_paragraph_result/ \
    --best_paragraph_file train_best_paragraph.json \
    --related_paragraph_file train_related_paragraph.json \
    --new_context_file train_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 128
echo "predict train first hop result done!"
echo "start predict dev first hop result !"
python -u first_hop_selector_predictor.py \
    --bert_model roberta-large \
    --checkpoint_path ../../data/checkpoints/selector/roberta_first_hop_selector \
    --model_name RobertaForRelatedSentence \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --predict_result_path ../../data/selector/roberta_first_hop_related_paragraph_result/ \
    --best_paragraph_file dev_best_paragraph.json \
    --related_paragraph_file dev_related_paragraph.json \
    --new_context_file dev_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 128
echo "predict dev first hop result done!"
echo "----------------------------------------------------"



echo "----------------------------------------------------"
echo "start train second hop selector..."
python -u second_hop_selector.py \
    --bert_model roberta-large \
    --over_write_result True \
    --output_dir ../../data/checkpoints/selector/roberta_second_paragraph_hop_selector \
    --feature_cache_path ../../data/cache/selector/roberta_second_paragraph_hop_selector \
    --model_name RobertaForRelatedSentence \
    --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --first_predict_result_path ../../data/selector/roberta_first_hop_related_paragraph_result/ \
    --best_paragraph_file train_best_paragraph.json \
    --related_paragraph_file train_related_paragraph.json \
    --new_context_file train_new_context.json \
    --dev_best_paragraph_file dev_best_paragraph.json \
    --dev_related_paragraph_file dev_related_paragraph.json \
    --dev_new_context_file dev_new_context.json \
    --output_log ../log/selector_two_hop.txt \
    --use_file_cache True \
    --max_seq_length 512 \
    --train_batch_size 12 \
    --val_batch_size 128 \
    --save_model_step 10000 \
    --num_train_epochs 3.0
echo "----------------------------------------------------"
echo "train second hop selector done!"


echo "----------------------------------------------------"
echo "start predict second hop result..."
python -u second_hop_selector_predictor.py \
    --bert_model bert-base-uncased \
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
    --bert_model bert-base-uncased \
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