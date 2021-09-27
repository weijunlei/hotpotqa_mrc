# 下载Hotpot数据
if [ ! -d "../../data/hotpot_data/hotpot_train_v1.1.json" ];then
  mkdir -p ../../data/hotpot_data
  cd ../../data/hotpot_data
  wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
  wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
  else
    echo "文件已存在"
fi
# 下载Spacy数据，对名词类词汇进行标准化
echo "download spacy models"
python -m spacy download en