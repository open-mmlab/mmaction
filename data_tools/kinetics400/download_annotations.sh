#! /usr/bin/bash env

DATA_DIR="../../data/kinetics400/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

wget https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz
tar -xf kinetics400.tar.gz -C ${DATA_DIR}/
rm kinetics400.tar.gz
