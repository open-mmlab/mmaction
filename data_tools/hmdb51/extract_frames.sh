#! /usr/bin/bash env

num_gpu=($(nvidia-smi -L | wc -l))
num_worker=${num_gpu}

cd ../
python build_rawframes.py ../data/hmdb51/videos/ ../data/hmdb51/rawframes/ --level 2 --flow_type tvl1 --num_gpu ${num_gpu} --num_worker ${num_worker}
echo "Raw frames (RGB and tv-l1) Generated"

cd hmdb51/
