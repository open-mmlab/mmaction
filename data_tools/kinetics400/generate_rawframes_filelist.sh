#! /usr/bin/bash env

cd ../../
PYTHONPATH=. python data_tools/build_file_list.py kinetics400 data/kinetics400/rawframes_train/ --level 2 --format rawframes --num_split 1 --subset train --shuffle
echo "Train filelist for rawframes generated."

PYTHONPATH=. python data_tools/build_file_list.py kinetics400 data/kinetics400/rawframes_val/ --level 2 --format rawframes --num_split 1 --subset val --shuffle
echo "Val filelist for rawframes generated."
cd data_tools/kinetics400/