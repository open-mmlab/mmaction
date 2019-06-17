#! /usr/bin/bash env

cd ../../
PYTHONPATH=. python data_tools/build_file_list.py kinetics400 data/kinetics400/videos_train/ --level 2 --format videos --num_split 1 --subset train --shuffle
echo "Train filelist for video generated."

PYTHONPATH=. python data_tools/build_file_list.py kinetics400 data/kinetics400/videos_val/ --level 2 --format videos --num_split 1 --subset val --shuffle
echo "Val filelist for video generated."
cd data_tools/kinetics400/