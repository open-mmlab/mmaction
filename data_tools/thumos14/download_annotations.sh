#! /usr/bin/bash env

DATA_DIR="../../data/thumos14/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

wget http://crcv.ucf.edu/THUMOS14/Validation_set/TH14_Temporal_annotations_validation.zip
wget http://crcv.ucf.edu/THUMOS14/test_set/TH14_Temporal_annotations_test.zip

unzip -j TH14_Temporal_annotations_validation.zip -d $DATA_DIR/annotations_val
unzip -j TH14_Temporal_annotations_test.zip -d $DATA_DIR/annotations_test