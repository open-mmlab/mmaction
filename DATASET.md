## Dataset Preparation

### Notes on Video Data format
MMAction supports two types of data format: raw frames and video. The former is widely used in previous projects such as [TSN](https://github.com/yjxiong/temporal-segment-networks).
This is fast (especially when SSD is available) but fails to scale to the fast-growing datasets.
(For example, the newest edition of [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) has 650K  videos and the total frames will take up several TBs.)
The latter save much space but is slower due to video decoding at execution time.
To alleviate such issue, we use [decord](https://github.com/zhreshold/decord) for efficient video loading.

For action recognition, both formats are supported.
For temporal action detection and spatial-temporal action detection, we still recommend the format of raw frames.

**TL;DR** We provide shell scripts for data preparation under the path `$MMACTION/data_tools/`.

### Prepare annotations
Please first refer to the official website (links attached) to fetch the annotations.
Supported datasets include but are not limited to:

- [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)
- [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)
- [THUMOS14](https://www.crcv.ucf.edu/THUMOS14/download.html)
- [AVA](https://research.google.com/ava/)


### Prepare videos
Please refer to the official website and/or the official script to prepare the videos.
Note that the videos should be arranged in either (1) a two-level directory organized by `${CLASS_NAME}/${VIDEO_ID}` or (2) a single-level directory.
It is recommended using (1) for action recognition datasets (such as UCF101 and Kinetics) and using (2) for action detection datasets or those with multiple annotations per video (such as THUMOS14 and AVA).

In the context of the whole project, the folder structure will look like:
```
mmaction
├── mmaction
├── tools
├── configs
├── data
│   ├── ucf101
│   │   ├── annotations
│   │   ├── videos
│   │   │   ├── ApplyEyeMakeup
│   │   │   │   ├── v_ApplyEyeMakeup_g01_c01.avi

│   │   │   ├── YoYo
│   │   │   │   ├── v_YoYo_g25_c05.avi
│   │   ├── rawframes
│   │   │   ├── ApplyEyeMakeup
│   │   │   │   ├── v_ApplyEyeMakeup_g01_c01
│   │   │   │   │   ├── 

│   │   │   ├── YoYo
│   │   │   │   ├── v_YoYo_g25_c05
│   │   │   │   │   ├── 


│   ├── thumos14
│   │   ├── annotations_val
│   │   ├── annotations_test
│   │   ├── videos_val
│   │   │   ├── video_validation_0000001.mp4

│   │   ├── rawframes_val
│   │   │   ├── video_validation_0000001

```

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
    - "$MMACTION/data/$DATASET/rawframes" `--format rawframes`
    - "$MMACTION/data/$DATASET/videos" if `--format videos`
