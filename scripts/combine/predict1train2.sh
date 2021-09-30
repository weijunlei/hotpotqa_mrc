echo "----------------------------------------------------"
echo "start predict second hop result..."
export CUDA_VISIBLE_DEVICES="0,1"
#cd ../../src/selector
#echo "start predict train first hop result !"
#python -u first_hop_selector_predictor.py \
#    --bert_model bert-base-uncased \
#    --checkpoint_path ../../data/checkpoints/selector/first_hop_just_paragraph_selector \
#    --model_name BertForParagraphClassification \
#    --dev_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
#    --predict_result_path ../../data/selector/first_hop_just_paragraph_result/ \
#    --best_paragraph_file train_best_paragraph.json \
#    --related_paragraph_file train_related_paragraph.json \
#    --new_context_file train_new_context.json \
#    --max_seq_length 512 \
#    --val_batch_size 64
#echo "predict train first hop result done!"
#echo "start predict dev first hop result !"
#python -u first_hop_selector_predictor.py \
#    --bert_model bert-base-uncased \
#    --checkpoint_path ../../data/checkpoints/selector/first_hop_just_paragraph_selector \
#    --model_name BertForParagraphClassification \
#    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
#    --predict_result_path ../../data/selector/first_hop_just_paragraph_result/ \
#    --best_paragraph_file dev_best_paragraph.json \
#    --related_paragraph_file dev_related_paragraph.json \
#    --new_context_file dev_new_context.json \
#    --max_seq_length 512 \
#    --val_batch_size 64
#echo "predict dev first hop result done!"
#echo "----------------------------------------------------"


#echo "----------------------------------------------------"
#echo "start train second hop selector..."
#cd ../../src/selector
#python -u second_hop_selector.py \
#    --bert_model bert-base-uncased \
#    --over_write_result True \
#    --output_dir ../../data/checkpoints/selector/second_hop_just_paragraph_hop_selector \
#    --feature_cache_path ../../data/cache/selector/second_hop_just_paragraph_selector \
#    --model_name BertForParagraphClassification \
#    --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
#    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
#    --first_predict_result_path ../../data/selector/first_hop_just_paragraph_result/ \
#    --best_paragraph_file train_best_paragraph.json \
#    --related_paragraph_file train_related_paragraph.json \
#    --new_context_file train_new_context.json \
#    --dev_best_paragraph_file dev_best_paragraph.json \
#    --dev_related_paragraph_file dev_related_paragraph.json \
#    --dev_new_context_file dev_new_context.json \
#    --output_log ../log/selector_2_base_2e-5.txt \
#    --use_file_cache True \
#    --max_seq_length 512 \
#    --train_batch_size 16 \
#    --val_batch_size 64 \
#    --save_model_step 50000 \
#    --num_train_epochs 3.0
#echo "----------------------------------------------------"
#echo "train second hop selector done!"



echo "----------------------------------------------------"
echo "start predict second hop result..."
cd ../../src/selector
python -u second_hop_selector_predictor.py \
    --bert_model bert-base-uncased \
    --checkpoint_path ../../data/checkpoints/selector/second_hop_just_paragraph_hop_selector \
    --model_name BertForParagraphClassification \
    --dev_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --first_predict_result_path ../../data/selector/first_hop_related_paragraph_result/ \
    --second_predict_result_path ../../data/selector/second_hop_related_paragraph_result/ \
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
    --checkpoint_path ../../data/checkpoints/selector/second_paragraph_hop_selector \
    --model_name BertForParagraphClassification \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --first_predict_result_path ../../data/selector/first_hop_just_paragraph_result/ \
    --second_predict_result_path ../../data/selector/second_hop_related_paragraph_result/ \
    --final_related_result dev_related.json \
    --best_paragraph_file dev_best_paragraph.json \
    --related_paragraph_file dev_related_paragraph.json \
    --new_context_file dev_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 128
echo "predict dev second hop result done!"
echo "----------------------------------------------------"