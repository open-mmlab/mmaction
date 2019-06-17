#! /usr/bin/bash env

DATA_DIR="../../data/thumos14/"

cd ${DATA_DIR}

wget https://storage.googleapis.com/thumos14_files/TH14_validation_set_mp4.zip
wget https://storage.googleapis.com/thumos14_files/TH14_Test_set_mp4.zip

unzip -j TH14_validation_set_mp4.zip -d videos_val/

unzip -P "THUMOS14_REGISTERED" TH14_Test_set_mp4.zip -d videos_test/

cd ../../data_tools/thumos14/