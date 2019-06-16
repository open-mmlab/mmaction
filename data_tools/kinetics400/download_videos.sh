#! /usr/bin/bash env

cd ../../mmaction/third_party/ActivityNet/Crawler/Kinetics

# set up environment
conda env create -f environment.yml
source activate kinetics
pip install --upgrade youtube-dl

ANNO_DIR="../../../../../data/kinetics400/annotations"
VIDEO_DIR="../../../../../data/kinetics400/videos"
mkdir -p ${VIDEO_DIR}
python download.py ${ANNO_DIR}/kinetics_train.csv ${VIDEO_DIR}/train
python download.py ${ANNO_DIR}/kinetics_val.csv ${VIDEO_DIR}/val
