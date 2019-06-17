#! /usr/bin/bash env

wget -c https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt -P ../../data/ava/annotations/


cat ../../data/ava/annotations/ava_file_names_trainval_v2.1.txt | while read vid; do wget -c "https://s3.amazonaws.com/ava-dataset/trainval/${vid}" -P ../../data/ava/videos_trainval/; done

echo "Downloading finished."