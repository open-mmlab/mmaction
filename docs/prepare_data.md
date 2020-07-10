# Preparing ActivityNet

For basic dataset information, please refer to the official [website](http://activity-net.org/).
Here, we use the ActivityNet rescaled feature provided in this [repo](https://github.com/wzmsltw/BSN-boundary-sensitive-network#code-and-data-preparation).
Before we start, please make sure that current working directory is `$MMACTION/tools/data/activitynet/`.

## Step 1. Download Annotations
First of all, you can run the following script to download annotation files.
```shell
bash download_annotations.sh
```

## Step 2. Prepare Videos Features
Then, you can run the following script to download activitynet features.
```shell
bash download_features.sh
```

## Step 3. Process Annotation Files
Next, you can run the following script to process the downloaded annotation files for training and testing.
It first merges the two annotation files together and then seperates the annoations by `train`, `val` and `test`.

```shell
python process_annotations.py
```

## Step 4. Check Directory Structure

After the whole data pipeline for ActivityNet preparation,
you will get the features and annotation files.

In the context of the whole project (for ActivityNet only), the folder structure will look like:

```
mmaction
├── mmaction
├── tools
├── configs
├── data
│   ├── ActivityNet
│   │   ├── anet_anno_{train,val,test,full}.json
│   │   ├── anet_anno_action.json
│   │   ├── video_info_new.csv
│   │   ├── activitynet_feature_cuhk
│   │   │   ├── csv_mean_100
│   │   │   │   ├── v___c8enCfzqw.csv
│   │   │   │   ├── v___dXUJsj3yo.csv
│   │   │   |   ├── ..
```

For training and evaluating on ActivityNet, please refer to [getting_started.md](/docs/getting_started.md).
# Preparing Kinetics-400

For basic dataset information, please refer to the official [website](https://deepmind.com/research/open-source/open-source-datasets/kinetics/).
Before we start, please make sure that the directory is located at `$MMACTION/tools/data/kinetics400/`.

## Step 1. Prepare Annotations

First of all, you can run the following script to prepare annotations.

```shell
bash download_annotations.sh
```

## Step 2. Prepare Videos

Then, you can run the following script to prepare videos.
The codes are adapted from the [official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics). Note that this might take a long time.

```shell
bash download_videos.sh
```

If you have already have a backup of the kinetics-400 dataset using the download script above,
you only need to replace all whitespaces in the class name for ease of processing either by [detox](http://manpages.ubuntu.com/manpages/bionic/man1/detox.1.html)

```shell
# sudo apt-get install detox
detox -r ../../../data/kinetics400/videos_train/
detox -r ../../../data/kinetics400/videos_val/
```

or running

```shell
bash rename_classnames.sh
```

## Step 3. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [dense_flow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance. And you can run the following script to soft link the extracted frames.

```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/kinetics400_extracted_train/
ln -s /mnt/SSD/kinetics400_extracted_train/ ../../../data/kinetics400/rawframes_train/
mkdir /mnt/SSD/kinetics400_extracted_val/
ln -s /mnt/SSD/kinetics400_extracted_val/ ../../../data/kinetics400/rawframes_val/
```

If you didn't install dense_flow in the installation or only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames.

```shell
bash extract_rgb_frames.sh
```

If both are required, run the following script to extract frames.

```shell
bash extract_frames.sh
```

These two commands above can generate images with size 340x256, if you want to generate images with short edge 320 (320p),
you can change the args `--new-width 340 --new-height 256` to `--new-short 320`.
More details can be found in [data_preparation](/docs/data_preparation.md)

## Step 4. Generate File List

you can run the follow scripts to generate file list in the format of videos and rawframes, respectively.

```shell
bash generate_videos_filelist.sh
# execute the command below when rawframes are ready
bash generate_rawframes_filelist.sh
```

## Step 5. Folder Structure

After the whole data pipeline for Kinetics-400 preparation.
you can get the rawframes (RGB + Flow), videos and annotation files for Kinetics-400.

In the context of the whole project (for Kinetics-400 only), the *minimal* folder structure will look like:
(*minimal* means that some data are not necessary: for example, you may want to evaluate kinetics-400 using the original video format.)

```
mmaction
├── mmaction
├── tools
├── configs
├── data
│   ├── kinetics400
│   │   ├── kinetics400_train_list_videos.txt
│   │   ├── kinetics400_val_list_videos.txt
│   │   ├── annotations
│   │   ├── videos_train
│   │   ├── videos_val
│   │   │   ├── abseiling
│   │   │   │   ├── 0wR5jVB-WPk_000417_000427.mp4
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   │   ├── wrapping_present
│   │   │   ├── ...
│   │   │   ├── zumba
│   │   ├── rawframes_train
│   │   ├── rawframes_val

```

For training and evaluating on Kinetics-400, please refer to [getting_started](/docs/getting_started.md).
# Preparing Moments in Time

For basic dataset information, you can refer to the dataset [website](http://moments.csail.mit.edu/).
Before we start, please make sure that the directory is located at `$MMACTION/tools/data/mit/`.

## Step 1. Prepare Annotations and Videos

First of all, you can run the following script to download the videos along with the annotations.

```shell
bash download_data.sh
```

## Step 2. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [dense_flow](https://github.com/open-mmlab/denseflow).


If you didn't install dense_flow in the installation or only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames.

Fist, You can run the following script to soft link the extracted frames.

```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/mit_extracted/
ln -s /mnt/SSD/mit_extracted/ ../../../data/mit/rawframes
```

```shell
bash extract_rgb_frames.sh
```

If both are required, run the following script to extract frames.

```shell
bash extract_frames.sh
```

## Step 4. Generate File List

you can run the follow script to generate file list in the format of rawframes and videos.

```shell
bash generate_{rawframes, videos}_filelist.sh
```

## Step 5. Check Directory Structure

After the whole data process for Moments in Time preparation,
you will get the rawframes (RGB + Flow), videos and annotation files for Moments in Time.

In the context of the whole project (for Moments in Time only), the folder structure will look like:

```
mmaction
├── data
│   └── mit
│       ├── annotations
│       │   ├── license.txt
│       │   ├── moments_categories.txt
│       │   ├── README.txt
│       │   ├── trainingSet.csv
│       │   └── validationSet.csv
│       ├── mit_train_rawframe_anno.txt
│       ├── mit_train_video_anno.txt
│       ├── mit_val_rawframe_anno.txt
│       ├── mit_val_video_anno.txt
│       ├── rawframes
│       │   ├── training
│       │   │   ├── adult+female+singing
│       │   │   │   ├── 0P3XG_vf91c_35
│       │   │   │   │   ├── flow_x_00001.jpg
│       │   │   │   │   ├── flow_x_00002.jpg
│       │   │   │   │   ├── ...
│       │   │   │   │   ├── flow_y_00001.jpg
│       │   │   │   │   ├── flow_y_00002.jpg
│       │   │   │   │   ├── ...
│       │   │   │   │   ├── img_00001.jpg
│       │   │   │   │   └── img_00002.jpg
│       │   │   │   └── yt-zxQfALnTdfc_56
│       │   │   │   │   ├── ...
│       │   │   └── yawning
│       │   │       ├── _8zmP1e-EjU_2
│       │   │       │   ├── ...
│       │   └── validation
│       │   │       ├── ...
│       └── videos
│           ├── training
│           │   ├── adult+female+singing
│           │   │   ├── 0P3XG_vf91c_35.mp4
│           │   │   ├── ...
│           │   │   └── yt-zxQfALnTdfc_56.mp4
│           │   └── yawning
│           │       ├── ...
│           └── validation
│           │   ├── ...
└── mmaction
└── ...

```

For training and evaluating on Moments in Time, please refer to [getting_started.md](/docs/getting_started.md).
# Preparing Multi-Moments in Time

For basic dataset information, you can refer to the dataset [website](moments.csail.mit.edu).
Before we start, please make sure that the directory is located at `$MMACTION/tools/data/mmit/`.

## Step 1. Prepare Annotations and Videos

First of all, you can run the following script to prepare annotations.

```shell
bash download_data.sh
```

## Step 2. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [dense_flow](https://github.com/open-mmlab/denseflow).

First, you can run the following script to soft link SSD.

```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/mmit_extracted/
ln -s /mnt/SSD/mmit_extracted/ ../../../data/mmit/rawframes
```

If you didn't install dense_flow in the installation or only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames.

```shell
bash extract_rgb_frames.sh
```

If both are required, run the following script to extract frames using "tvl1" algorithm.

```shell
bash extract_frames.sh
```
## Step 3. Generate File List

you can run the follow script to generate file list in the format of rawframes or videos.

```shell
bash generate_rawframes_filelist.sh
bash generate_videos_filelist.sh
```

## Step 4. Check Directory Structure

After the whole data process for Multi-Moments in Time preparation,
you will get the rawframes (RGB + Flow), videos and annotation files for Multi-Moments in Time.

In the context of the whole project (for Multi-Moments in Time only), the folder structure will look like:

```
mmaction/
└── data
    └── mmit
        ├── annotations
        │   ├── moments_categories.txt
        │   ├── trainingSet.txt
        │   └── validationSet.txt
        ├── mmit_train_rawframes.txt
        ├── mmit_train_videos.txt
        ├── mmit_val_rawframes.txt
        ├── mmit_val_videos.txt
        ├── rawframes
        │   ├── 0-3-6-2-9-1-2-6-14603629126_5
        │   │   ├── flow_x_00001.jpg
        │   │   ├── flow_x_00002.jpg
        │   │   ├── ...
        │   │   ├── flow_y_00001.jpg
        │   │   ├── flow_y_00002.jpg
        │   │   ├── ...
        │   │   ├── img_00001.jpg
        │   │   └── img_00002.jpg
        │   │   ├── ...
        │   └── yt-zxQfALnTdfc_56
        │   │   ├── ...
        │   └── ...

        └── videos
            └── adult+female+singing
                ├── 0-3-6-2-9-1-2-6-14603629126_5.mp4
                └── yt-zxQfALnTdfc_56.mp4
            └── ...
```

For training and evaluating on Multi-Moments in Time, please refer to [getting_started.md](/docs/getting_started.md).
# Preparing Something-Something V1

For basic dataset information, you can refer to the dataset [website](https://20bn.com/datasets/something-something/v1).
Before we start, please make sure that the directory is located at `$MMACTION/tools/data/sthv1/`.

## Step 1. Prepare Annotations

First of all, you have to sign in and download annotations to `$MMACTION/data/sthv1/annotations` on the official [website](https://20bn.com/datasets/something-something/v1).

## Step 2. Prepare Videos

Then, you can download all data parts to `$MMACTION/data/sthv1/` and use the following command to extract.

```shell
cd $MMACTION/data/sthv1/
cat 20bn-something-something-v1-?? | tar zx
cd $MMACTION/tools/data/sthv1/
```

## Step 3. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [dense_flow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance.

You can run the following script to soft link SSD.

```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/sthv1_extracted/
ln -s /mnt/SSD/sthv1_extracted/ ../../../data/sthv1/rawframes
```

If you didn't install dense_flow in the installation or only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames.

```shell
cd $MMACTION/tools/data/sthv1/
bash extract_rgb_frames.sh
```

If both are required, run the following script to extract frames.

```shell
cd $MMACTION/tools/data/sthv1/
bash extract_frames.sh
```

## Step 4. Generate File List

you can run the follow script to generate file list in the format of rawframes and videos.

```shell
cd $MMACTION/tools/data/sthv1/
bash generate_{rawframes, videos}_filelist.sh
```

## Step 5. Check Directory Structure

After the whole data process for Something-Something V1 preparation,
you will get the rawframes (RGB + Flow), videos and annotation files for Something-Something V1.

In the context of the whole project (for Something-Something V1 only), the folder structure will look like:

```
mmaction
├── mmaction
├── tools
├── configs
├── data
│   ├── sthv1
│   │   ├── sthv1_{train,val}_list_rawframes.txt
│   │   ├── sthv1_{train,val}_list_videos.txt
│   │   ├── annotations
│   |   ├── videos
│   |   |   ├── 100000.mp4
│   |   |   ├── 100001.mp4
│   |   |   ├──...
│   |   ├── rawframes
│   |   |   ├── 100000
│   |   |   |   ├── img_00001.jpg
│   |   |   |   ├── img_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_x_00001.jpg
│   |   |   |   ├── flow_x_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_y_00001.jpg
│   |   |   |   ├── flow_y_00002.jpg
│   |   |   |   ├── ...
│   |   |   ├── 100001
│   |   |   ├── ...

```

For training and evaluating on Something-Something V1, please refer to [getting_started.md](/docs/getting_started.md).
# Preparing Something-Something V2

For basic dataset information, you can refer to the dataset [website](https://20bn.com/datasets/something-something/v2).
Before we start, please make sure that the directory is located at `$MMACTION/tools/data/sthv2/`.

## Step 1. Prepare Annotations

First of all, you have to sign in and download annotations to `$MMACTION/data/sthv2/annotations` on the official [website](https://20bn.com/datasets/something-something/v2).

## Step 2. Prepare Videos

Then, you can download all data parts to `$MMACTION/data/sthv2/` and use the following command to extract.

```shell
cd $MMACTION/data/sthv2/
cat 20bn-something-something-v2-?? | tar zx
```

## Step 3. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [dense_flow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance.

You can run the following script to soft link SSD.

```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/sthv2_extracted/
ln -s /mnt/SSD/sthv2_extracted/ ../../../data/sthv2/rawframes
```

If you didn't install dense_flow in the installation or only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames.

```shell
cd $MMACTION/tools/data/sthv2/
bash extract_rgb_frames.sh
```

If both are required, run the following script to extract frames.

```shell
cd $MMACTION/tools/data/sthv2/
bash extract_frames.sh
```

## Step 4. Generate File List

you can run the follow script to generate file list in the format of rawframes and videos.

```shell
cd $MMACTION/tools/data/sthv2/
bash generate_{rawframes, videos}_filelist.sh
```

## Step 5. Check Directory Structure

After the whole data process for Something-Something V2 preparation,
you will get the rawframes (RGB + Flow), videos and annotation files for Something-Something V2.

In the context of the whole project (for Something-Something V2 only), the folder structure will look like:

```
mmaction
├── mmaction
├── tools
├── configs
├── data
│   ├── sthv2
│   │   ├── sthv2_{train,val}_list_rawframes.txt
│   │   ├── sthv2_{train,val}_list_videos.txt
│   │   ├── annotations
│   |   ├── videos
│   |   |   ├── 100000.mp4
│   |   |   ├── 100001.mp4
│   |   |   ├──...
│   |   ├── rawframes
│   |   |   ├── 100000
│   |   |   |   ├── img_00001.jpg
│   |   |   |   ├── img_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_x_00001.jpg
│   |   |   |   ├── flow_x_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_y_00001.jpg
│   |   |   |   ├── flow_y_00002.jpg
│   |   |   |   ├── ...
│   |   |   ├── 100001
│   |   |   ├── ...

```

For training and evaluating on Something-Something V2, please refer to [getting_started.md](/docs/getting_started.md).
# Preparing THUMOS'14

For basic dataset information, you can refer to the dataset [website](https://www.crcv.ucf.edu/THUMOS14/download.html).
Before we start, please make sure that the directory is located at `$MMACTION/tools/data/thumos14/`.

## Step 1. Prepare Annotations

First of all, run the following script to prepare annotations.

```shell
cd $MMACTION/tools/data/thumos14/
bash download_annotations.sh
```

## Step 2. Prepare Videos

Then, you can run the following script to prepare videos.

```shell
cd $MMACTION/tools/data/thumos14/
bash download_videos.sh
```

## Step 3. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [dense_flow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance.

You can run the following script to soft link SSD.

```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/thumos14_extracted/
ln -s /mnt/SSD/thumos14_extracted/ ../data/thumos14/rawframes/
```

If you didn't install dense_flow in the installation or only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames.

```shell
cd $MMACTION/tools/data/thumos14/
bash extract_rgb_frames.sh
```

If both are required, run the following script to extract frames.

```shell
cd $MMACTION/tools/data/thumos14/
bash extract_frames.sh tvl1
```

## Step 4. Fetch File List

You can run the follow script to fetch pre-computed tag proposals.

```shell
cd $MMACTION/tools/data/thumos14/
bash fetch_tag_proposals.sh
```

## Step 5. Check Directory Structure

After the whole data process for THUMOS'14 preparation,
you will get the rawframes (RGB + Flow), videos and annotation files for THUMOS'14.

In the context of the whole project (for THUMOS'14 only), the folder structure will look like:

```
mmaction
├── mmaction
├── tools
├── configs
├── data
│   ├── thumos14
│   │   ├── proposals
│   │   |   ├── thumos14_tag_val_normalized_proposal_list.txt
│   │   |   ├── thumos14_tag_test_normalized_proposal_list.txt
│   │   ├── annotations_val
│   │   ├── annotations_test
│   │   ├── videos
│   │   │   ├── val
│   │   │   |   ├── video_validation_0000001.mp4
│   │   │   |   ├── ...
│   │   |   ├── test
│   │   │   |   ├── video_test_0000001.mp4
│   │   │   |   ├── ...
│   │   ├── rawframes
│   │   │   ├── val
│   │   │   |   ├── video_validation_0000001
|   │   │   |   │   ├── img_00001.jpg
|   │   │   |   │   ├── img_00002.jpg
|   │   │   |   │   ├── ...
|   │   │   |   │   ├── flow_x_00001.jpg
|   │   │   |   │   ├── flow_x_00002.jpg
|   │   │   |   │   ├── ...
|   │   │   |   │   ├── flow_y_00001.jpg
|   │   │   |   │   ├── flow_y_00002.jpg
|   │   │   |   │   ├── ...
│   │   │   |   ├── ...
│   │   |   ├── test
│   │   │   |   ├── video_test_0000001
```

For training and evaluating on THUMOS'14, please refer to [getting_started.md](/docs/getting_started.md).
# Preparing UCF-101

For basic dataset information, you can refer to the dataset [website](https://www.crcv.ucf.edu/data/UCF101.php).
Before we start, please make sure that the directory is located at `$MMACTION/tools/data/ucf101/`.

## Step 1. Prepare Annotations

First of all, you can run the following script to prepare annotations.

```shell
bash download_annotations.sh
```

## Step 2. Prepare Videos

Then, you can run the following script to prepare videos.

```shell
bash download_videos.sh
```

## Step 3. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [dense_flow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance. The extracted frames (RGB + Flow) will take up about 100GB.

You can run the following script to soft link SSD.

```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/ucf101_extracted/
ln -s /mnt/SSD/ucf101_extracted/ ../../../data/ucf101/rawframes
```

If you didn't install dense_flow in the installation or only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames.

```shell
bash extract_rgb_frames.sh
```

If both are required, run the following script to extract frames using "tvl1" algorithm.

```shell
bash extract_frames.sh
```

## Step 4. Generate File List

you can run the follow script to generate file list in the format of rawframes and videos.

```shell
bash generate_{rawframes, videos}_filelist.sh
```

## Step 5. Check Directory Structure

After the whole data process for UCF-101 preparation,
you will get the rawframes (RGB + Flow), videos and annotation files for UCF-101.

In the context of the whole project (for UCF-101 only), the folder structure will look like:

```
mmaction
├── mmaction
├── tools
├── configs
├── data
│   ├── ucf101
│   │   ├── ucf101_{train,val}_split_{1,2,3}_rawframes.txt
│   │   ├── ucf101_{train,val}_split_{1,2,3}_videos.txt
│   │   ├── annotations
│   │   ├── videos
│   │   │   ├── ApplyEyeMakeup
│   │   │   │   ├── v_ApplyEyeMakeup_g01_c01.avi

│   │   │   ├── YoYo
│   │   │   │   ├── v_YoYo_g25_c05.avi
│   │   ├── rawframes
│   │   │   ├── ApplyEyeMakeup
│   │   │   │   ├── v_ApplyEyeMakeup_g01_c01
│   │   │   │   │   ├── img_00001.jpg
│   │   │   │   │   ├── img_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_x_00001.jpg
│   │   │   │   │   ├── flow_x_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_y_00001.jpg
│   │   │   │   │   ├── flow_y_00002.jpg
│   │   │   ├── ...
│   │   │   ├── YoYo
│   │   │   │   ├── v_YoYo_g01_c01
│   │   │   │   ├── ...
│   │   │   │   ├── v_YoYo_g25_c05

```

For training and evaluating on UCF-101, please refer to [getting_started.md](/docs/getting_started.md).
