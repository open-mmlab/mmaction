#! /usr/bin/bash env

cd ../
python build_rawframes.py ../data/ucf101/videos/ ../data/ucf101/rawframes/ --level 2
echo "Raw frames (RGB only) Generated"
cd ucf101/
