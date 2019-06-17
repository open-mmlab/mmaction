#! /usr/bin/bash env

wget -c https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt -P ../../data/ava/annotations/


# sudo apt-get install parallel
# parallel downloading to speed up
awk '{print "https://s3.amazonaws.com/ava-dataset/trainval/"$0}' ../../data/ava/annotations/ava_file_names_trainval_v2.1.txt | parallel -j 8 wget -c -q {} -P ../../data/ava/videos_trainval/
echo "Parallel downloading finished."