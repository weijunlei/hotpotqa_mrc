echo "----------------------------------------------------"
echo "start predict first hop result..."
# BertForParagraphClassification
# BertForRelatedSentence
export CUDA_VISIBLE_DEVICES="3"
MODEL_NAME=BertForRelatedSentence
cd ../../src/selector
echo "start predict dev first hop result !"
#python -u first_hop_selector_predictor.py \
#    --bert_model bert-base-uncased \
#    --checkpoint_path ../../data/checkpoints/selector/20211214_first_hop_just_paragraph_selector_12 \
#    --model_name $MODEL_NAME \
#    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
#    --predict_result_path ../../data/selector/20211214_first_hop_just_paragraph_selector_12_result/ \
#    --best_paragraph_file dev_best_paragraph.json \
#    --related_paragraph_file dev_related_paragraph.json \
#    --new_context_file dev_new_context.json \
#    --max_seq_length 512 \
#    --val_batch_size 256
#echo "predict dev first hop result done!"
#echo "start send email"
#cd ../../src/preprocess
#python send_email.py 20211213_first_hop_related_paragraph_selector predict完成
echo "----------------------------------------------------"
echo "start predict train first hop result !"
cd ../../src/selector
python -u first_hop_selector_predictor.py \
    --bert_model bert-base-uncased \
    --checkpoint_path ../../data/checkpoints/selector/20211214_first_hop_related_paragraph_selector_12_split \
    --model_name $MODEL_NAME \
    --dev_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --predict_result_path ../../data/selector/20211214_first_hop_related_paragraph_selector_12_split_result/ \
    --best_paragraph_file train_best_paragraph.json \
    --related_paragraph_file train_related_paragraph.json \
    --new_context_file train_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 256
echo "predict train first hop result done!"
