#! /usr/bin/bash env

DATA_DIR="../../data/ava/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

wget https://research.google.com/ava/download/ava_v2.1.zip
unzip -j ava_v2.1.zip -d ${DATA_DIR}/
rm ava_v2.1.zip