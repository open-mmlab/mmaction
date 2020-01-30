#! /usr/bin/bash env

cd ../../data/ava/

ls ./videos_trimmed_trainval | while read filename; do \
  vid="$(echo ${filename} | cut -d'.' -f1)";
  resolution=`ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 ./videos_trimmed_trainval/${filename}`
  echo ${vid} ${resolution}
done &> ava_video_resolution_stats.csv

echo $PWD

cd ../../data_tools/ava/
