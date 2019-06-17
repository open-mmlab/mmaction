#! /usr/bin/bash env

cd ../
python build_rawframes.py ../data/ava/videos_trimmed_trainval/ ../data/ava/rawframes/ --level 1  --ext mp4
echo "Raw frames (RGB only) generated for train and val set"

cd ava/