export CUDA_VISIBLE_DEVICES=0
python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:7956' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0 ./ilsvrc2012/