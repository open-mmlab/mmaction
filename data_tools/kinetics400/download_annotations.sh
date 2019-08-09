#! /usr/bin/bash env

DATA_DIR="../../data/kinetics400/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

wget -P ${DATA_DIR} https://github.com/activitynet/ActivityNet/raw/master/Crawler/Kinetics/data/kinetics-400_train.csv
wget -P ${DATA_DIR} https://github.com/activitynet/ActivityNet/raw/master/Crawler/Kinetics/data/kinetics-400_val.csv
wget -P ${DATA_DIR} https://github.com/activitynet/ActivityNet/raw/master/Crawler/Kinetics/data/kinetics-400_test.csv
