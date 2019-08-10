#! /usr/bin/bash env

DATA_DIR="../../data/kinetics400/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

wget -O ${DATA_DIR}/kinetics_train.csv https://github.com/activitynet/ActivityNet/raw/master/Crawler/Kinetics/data/kinetics-400_train.csv
wget -O ${DATA_DIR}/kinetics_val.csv https://github.com/activitynet/ActivityNet/raw/master/Crawler/Kinetics/data/kinetics-400_val.csv
wget -O ${DATA_DIR}/kinetics_test.csv https://github.com/activitynet/ActivityNet/raw/master/Crawler/Kinetics/data/kinetics-400_test.csv
