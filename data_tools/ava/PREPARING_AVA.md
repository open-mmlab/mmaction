## Preparing AVA

For more details, please refer to the [official website](https://research.google.com/ava/). We provide scripts with documentations. Before we start, please make sure that the directory is located at `$MMACTION/data_tools/ava/`.

### Prepare annotations
First of all, run the following script to prepare annotations.
```shell
bash download_annotations.sh
```

### Prepare videos
Then, use the following script to prepare videos. The codes are adapted from the [official crawler](https://github.com/cvdfoundation/ava-dataset). Note that this might take a long time.
```shell
bash download_videos.sh
```
Note that if you happen to have sudoer or have [GNU parallel](https://www.gnu.org/software/parallel/) [<sup>1</sup>](#1) on your machine, you can speed up the procedure by downloading in parallel.

```shell
# sudo apt-get install parallel
bash download_videos_parallel.sh
```

### Preprocess videos
The videos vary in length, while the annotations are from 15min to 30min.
Therefore, we can preprocess videos to save storage and processing time afterward.
Run the following scripts to trim the videos into 17-min segments (from 00:14:00 to 00:31:00) with FPS adjusted to 30 FPS and height to be 480.

```shell
bash preprocess_videos.sh
```


### Extract frames
Now it is time to extract frames from videos. 
Before extraction, please refer to `DATASET.md` for installing [dense_flow](https://github.com/yjxiong/dense_flow).
If you have some SSD, then we strongly recommend extracting frames there for better I/O performance. 
```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/ava_extracted/
ln -s /mnt/SSD/ava_extracted/ ../data/ava/rawframes/
```
Afterwards, run the following script to extract frames.
```shell
bash extract_frames.sh
```
If you only want to play with RGB frames (since extracting optical flow can be both time-comsuming and space-hogging), consider running the following script to extract **RGB-only** frames.
```shell
bash extract_rgb_frames.sh
```


### Fetching proposal files and other metadata file
Run the follow scripts to fetch pre-computed proposal list.
The proposals are adapted from FAIR's [Long-Term Feature Banks](https://github.com/facebookresearch/video-long-term-feature-banks).
```shell
bash fetch_ava_proposals.sh
```
In addition, we use the following script to obtain the resolutions of all videos due to varying aspect ratio.
```shell
bash obtain_video_resolution.sh
```

### Folder structure
In the context of the whole project (for ava only), the folder structure will look like: 

```
mmaction
├── mmaction
├── tools
├── configs
├── data
│   ├── ava
│   │   ├── ava_video_resolution_stats.csv
│   │   ├── ava_dense_proposals_train.FAIR.recall_93.9.pkl
│   │   ├── ava_dense_proposals_val.FAIR.recall_93.9.pkl
│   │   ├── annotations
│   │   ├── videos_trainval
│   │   │   ├── 053oq2xB3oU.mkv
│   │   │   ├── 0f39OWEqJ24.mp4
│   │   │   ├── ...
│   │   ├── videos_trimmed_trainval
│   │   │   ├── 053oq2xB3oU.mp4
│   │   │   ├── 0f39OWEqJ24.mp4
│   │   │   ├── ...
│   │   ├── rawframes
│   │   │   ├── 053oq2xB3oU.mp4
|   │   │   │   ├── img_00001.jpg
|   │   │   │   ├── img_00002.jpg
|   │   │   │   ├── ...
```

For training and evaluating on AVA, please refer to [GETTING_STARTED.md](https://github.com/open-mmlab/mmaction/blob/master/GETTING_STARTED.md).


Reference

<a class="anchor" id="1">[1] O. Tange (2018): GNU Parallel 2018, March 2018, https://doi.org/10.5281/zenodo.1146014 </a>