#! /usr/bin/bash env

cd ../../data/ava/

mkdir ./videos_trimmed_trainval/
ls videos_trainval/ | while read filename; do \
  vid="$(echo ${filename} | cut -d'.' -f1)";
  ffmpeg -nostdin -i "./videos_trainval/${filename}" \
          -ss 00:14:00 -t 00:17:00 \
          -filter:v fps=fps=30 \
          "./${vid}.tmp.mp4";
  ffmpeg -nostdin -i "./${vid}.tmp.mp4" \
          -vf scale=-2:480 \
          -c:a copy \
          "./videos_trimmed_trainval/${vid}.mp4";
  rm "./${vid}.tmp.mp4";
done

cd ../../data_tools/ava/
