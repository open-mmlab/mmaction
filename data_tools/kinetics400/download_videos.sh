#! /usr/bin/bash env

cd ../../mmaction/third_party/ActivityNet/Crawler/Kinetics

# set up environment
conda env create -f environment.yml
source activate kinetics
pip install --upgrade youtube-dl

DATA_DIR="../../../../../data/kinetics400"
ANNO_DIR="../../../../../data/kinetics400/annotations"
python download.py ${ANNO_DIR}/kinetics400/train.csv ${DATA_DIR}/videos_train
python download.py ${ANNO_DIR}/kinetics400/val.csv ${DATA_DIR}/videos_val

cd ../../../../../data_tools/kinetics400
