#! /usr/bin/bash env

# Rename classname for convenience
cd ../../data/kinetics400/
ls ./videos_train | while read class; do \
  newclass=`echo $class | tr " " "_" | tr "(" "-" | tr ")" "-" `;
  if [ "${class}" != "${newclass}" ]
  then
    mv "videos_train/${class}" "videos_train/${newclass}";
  fi
done

ls ./videos_val | while read class; do \
  newclass=`echo $class | tr " " "_" | tr "(" "-" | tr ")" "-" `;
  if [ "${class}" != "${newclass}" ]
  then
    mv "videos_val/${class}" "videos_val/${newclass}";
  fi
done

cd ../../data_tools/kinetics400/
