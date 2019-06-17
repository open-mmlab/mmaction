#! /usr/bin/bash env

cd ../
python build_rawframes.py ../data/ava/videos_trimmed_trainval/ ../data/ava/rawframes/ --level 1 --flow_type tvl1 --ext mp4
echo "Raw frames (RGB and tv-l1) Generated for train+val set"

cd ava/
