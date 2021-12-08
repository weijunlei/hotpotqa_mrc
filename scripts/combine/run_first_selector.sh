echo "----------------------------------------------------"
export CUDA_VISIBLE_DEVICES="2"
FIRST_HOP_RESULT_NAME=20211208_first_hop_just_paragraph_selector
FIRST_HOP_PREDICT_NAME=20211208_first_hop_just_paragraph_result
MODEL_NAME=BertForParagraphClassification
echo "start train first hop selector..."
cd ../../src/selector
python -u first_hop_selector.py \
    --bert_model bert-base-uncased \
    --over_write_result True \
    --output_dir ../../data/checkpoints/selector/$FIRST_HOP_RESULT_NAME \
    --feature_cache_path ../../data/cache/selector/$FIRST_HOP_RESULT_NAME \
    --model_name $MODEL_NAME \
    --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --output_log ../log/selector_1_paragraph_base_2e-5.txt \
    --use_file_cache True \
    --max_seq_length 512 \
    --train_batch_size 12 \
    --val_batch_size 128 \
    --save_model_step 10000 \
    --num_train_epochs 3.0
echo "train first hop selector done!"
echo "----------------------------------------------------"

echo "----------------------------------------------------"
echo "start predict first hop result..."
echo "start predict train first hop result !"
python -u first_hop_selector_predictor.py \
    --bert_model bert-base-uncased \
    --checkpoint_path ../../data/checkpoints/selector/$FIRST_HOP_RESULT_NAME \
    --model_name $MODEL_NAME \
    --dev_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
    --predict_result_path ../../data/selector/$FIRST_HOP_RESULT_NAME/ \
    --best_paragraph_file train_best_paragraph.json \
    --related_paragraph_file train_related_paragraph.json \
    --new_context_file train_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 128
echo "predict train first hop result done!"
echo "start predict dev first hop result !"
python -u first_hop_selector_predictor.py \
    --bert_model bert-base-uncased \
    --checkpoint_path ../../data/checkpoints/selector/$$FIRST_HOP_RESULT_NAME \
    --model_name $MODEL_NAME \
    --dev_file ../../data/hotpot_data/hotpot_dev_labeled_data_v3.json \
    --predict_result_path ../../data/selector/$$FIRST_HOP_RESULT_NAME/ \
    --best_paragraph_file dev_best_paragraph.json \
    --related_paragraph_file dev_related_paragraph.json \
    --new_context_file dev_new_context.json \
    --max_seq_length 512 \
    --val_batch_size 128
echo "predict dev first hop result done!"
echo "----------------------------------------------------"