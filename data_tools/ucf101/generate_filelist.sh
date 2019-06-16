#! /usr/bin/bash env

cd ../../
PYTHONPATH=. python data_tools/build_file_list.py ucf101 data/ucf101/rawframes/ --level 2 --format rawframes
echo "Filelist Generated"
cd data_tools/ucf101/