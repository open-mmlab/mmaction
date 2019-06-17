#! /usr/bin/bash env

cd ../../mmaction/third_party/ActivityNet/Crawler/Kinetics

# set up environment
conda env create -f environment.yml
source activate kinetics
pip install --upgrade youtube-dl

DATA_DIR="../../../../../data/kinetics400"
ANNO_DIR="../../../../../data/kinetics400/annotations"
python download.py ${ANNO_DIR}/kinetics_train.csv ${DATA_DIR}/videos_train
python download.py ${ANNO_DIR}/kinetics_val.csv ${DATA_DIR}/videos_val

# Rename classname for convenience
cd ${DATA_DIR}
ls ./videos_train | while read class; do \
  newclass=`echo $class | tr " " "_"`;
  if [ $class != $newclass ]
  then
    mv "videos_train/${class}" "videos_train/${newclass}";
  fi
done

ls ./videos_val | while read class; do \
  newclass=`echo $class | tr " " "_"`;
  if [ $class != $newclass ]
  then
    mv "videos_val/${class}" "videos_val/${newclass}";
  fi
done

cd ../../data_tools/kinetics400/
