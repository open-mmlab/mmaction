#! /usr/bin/bash env

num_gpu=($(nvidia-smi -L | wc -l))
num_worker=${num_gpu}

cd ../
python build_rawframes.py ../data/hmdb51/videos/ ../data/hmdb51/rawframes/ --level 2  --ext avi --num_gpu ${num_gpu} --num_worker ${num_worker}
echo "Raw frames (RGB only) generated for train and val set"

cd hmdb51/