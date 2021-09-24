echo "----------------------------------------------------"
echo "start train reader model"
export CUDA_VISIBLE_DEVICES="0,1"
cd ../../src/reader
python -u origin_reader_model.py \
  --bert_model bert-base-uncased \
  --output_dir ../../data/checkpoints/qa_base_20210922_coattention \
  --model_name BertForQuestionAnsweringCoAttention \
  --train_file ../../data/hotpot_data/hotpot_train_labeled_data_v3.json \
  --dev_file ../../data/hotpot_data/hotpot_dev_distractor_v1.json \
  --train_filter_file ../../data/selector/second_hop_related_paragraph_result/train_related.json \
  --dev_filter_file ../../data/selector/second_hop_related_paragraph_result/dev_related.json \
  --use_file_cache \
  --feature_cache_path ../../data/cache/qa_base_20210922_coattention \
  --train_batch_size 16 \
  --val_batch_size 128 \
  --save_model_step 3000 \
  --num_train_epochs 2.0
echo "----------------------------------------------------"