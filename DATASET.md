## Dataset Preparation

### Notes on Video Data format
MMAction supports two types of data format: raw frames and video. The former is widely used in previous projects such as [TSN](https://github.com/yjxiong/temporal-segment-networks).
This is fast (especially when SSD is available) but fails to scale to the fast-growing datasets.
(For example, the newest edition of [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) has 650K  videos and the total frames will take up several TBs.)
The latter save much space but is slower due to video decoding at execution time.
To alleviate such issue, we use [decord](https://github.com/zhreshold/decord) for efficient video loading.

For action recognition, both formats are supported.
For temporal action detection and spatial-temporal action detection, we still recommend the format of raw frames.


### Supported datasets
The supported datasets are listed below.
We provide shell scripts for data preparation under the path `$MMACTION/data_tools/`.
To ease usage, we provide tutorials of data deployment for each dataset.

- [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/): See [PREPARING_HMDB51.md](https://github.com/open-mmlab/mmaction/tree/master/data_tools/hmdb51/PREPARING_HMDB51.md)
- [UCF101](https://www.crcv.ucf.edu/data/UCF101.php): See [PREPARING_UCF101.md](https://github.com/open-mmlab/mmaction/tree/master/data_tools/ucf101/PREPARING_UCF101.md)
- [Kinetics400](https://deepmind.com/research/open-source/open-source-datasets/kinetics/): See [PREPARING_KINETICS400.md](https://github.com/open-mmlab/mmaction/tree/master/data_tools/kinetics400/PREPARING_KINETICS400.md)
- [THUMOS14](https://www.crcv.ucf.edu/THUMOS14/download.html): See [PREPARING_TH14.md](https://github.com/open-mmlab/mmaction/tree/master/data_tools/thumos14/PREPARING_TH14.md)
- [AVA](https://research.google.com/ava/): See [PREPARING_AVA.md](https://github.com/open-mmlab/mmaction/tree/master/data_tools/ava/PREPARING_AVA.md)


Now, you can switch to [GETTING_STARTED.md](https://github.com/open-mmlab/mmaction/tree/master/GETTING_STARTED.md) to train and test the model.


**TL;DR** The following guide is helpful when you want to experiment with custom dataset.
Similar to the datasets stated above, it is recommended organizing in `$MMACTION/data/$DATASET`.

### Prepare annotations

### Prepare videos
Please refer to the official website and/or the official script to prepare the videos.
Note that the videos should be arranged in either (1) a two-level directory organized by `${CLASS_NAME}/${VIDEO_ID}` or (2) a single-level directory.
It is recommended using (1) for action recognition datasets (such as UCF101 and Kinetics) and using (2) for action detection datasets or those with multiple annotations per video (such as THUMOS14 and AVA).


### Extract frames
To extract frames (optical flow, to be specific), [dense_flow](https://github.com/yjxiong/dense_flow) is needed.
(**TODO**: This will be merged into MMAction in the next version in a smoother way).
For the time being, please use the following command:

```shell
python build_rawframes.py $SRC_FOLDER $OUT_FOLDER --df_path $PATH_OF_DENSE_FLOW --level {1, 2}
```
- `$SRC_FOLDER` points to the folder of the original video (for example)
- `$OUT_FOLDER` points to the root folder where the extracted frames and optical flow store 
- `$PATH_OF_DENSE_FLOW` points to the root folder where dense_flow is installed.
- `--level` is either 1 for the single-level directory or 2 for the two-level directory

The recommended practice is

1. set `$OUT_FOLDER` to be an folder located in SSD
2. symlink the link `$OUT_FOLDER` to `$MMACTION/data/$DATASET/rawframes`.

```shell
ln -s ${OUT_FOLDER} $MMACTION/data/$DATASET/rawframes
```

### Generate filelist
```shell
cd $MMACTION
python data_tools/build_file_list.py ${DATASET} ${SRC_FOLDER} --level {1, 2} --format {rawframes, videos}
```
- `${SRC_FOLDER}` should point to the folder of the corresponding to the data format:
    - "$MMACTION/data/$DATASET/rawframes" if `--format rawframes`
    - "$MMACTION/data/$DATASET/videos" if `--format videos`
