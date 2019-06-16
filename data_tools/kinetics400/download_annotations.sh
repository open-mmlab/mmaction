#! /usr/bin/bash env

DATA_DIR="../../data/kinetics400/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

wget https://deepmind.com/documents/66/kinetics_train.zip
wget https://deepmind.com/documents/65/kinetics_val.zip
wget https://deepmind.com/documents/81/kinetics_test.zip

unzip -j kinetics_train.zip -d ${DATA_DIR}/
unzip -j kinetics_val.zip -d ${DATA_DIR}/
unzip -j kinetics_test.zip -d ${DATA_DIR}/
rm kinetics_train.zip
rm kinetics_val.zip
rm kinetics_test.zip