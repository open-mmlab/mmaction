## Preparing Kinetics-400

For more details, please refer to the official [website](https://deepmind.com/research/open-source/open-source-datasets/kinetics/). We provide scripts with documentations. Before we start, please make sure that the directory is located at `$MMACTION/data_tools/kinetics400/`.

### Prepare annotations
First of all, run the following script to prepare annotations.
```shell
bash download_annotations.sh
```

### Prepare videos
Then, use the following script to prepare videos. The codes are adapted from the [official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics). Note that this might take a long time.
```shell
bash download_videos.sh
```
Note that some people may already have a backup of the kinetics-400 dataset using the [official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics).
If this is the case, then you only need to replace all whitespaces in the class name for ease of processing either by [detox](http://manpages.ubuntu.com/manpages/bionic/man1/detox.1.html)

```shell
# sudo apt-get install detox
detox -r ../../data/kinetics400/videos_train/
detox -r ../../data/kinetics400/videos_val/
```
or running 
```shell
bash rename_classnames.sh
```

### Extract frames
Now it is time to extract frames from videos. 
Before extraction, please refer to `DATASET.md` for installing [dense_flow](https://github.com/yjxiong/dense_flow).
If you have some SSD, then we strongly recommend extracting frames there for better I/O performance. 
```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/kinetics400_extracted_train/
ln -s /mnt/SSD/kinetics400_extracted_train/ ../data/kinetics400/rawframes_train/
mkdir /mnt/SSD/kinetics400_extracted_val/
ln -s /mnt/SSD/kinetics400_extracted_val/ ../data/kinetics400/rawframes_val/
```
Afterwards, run the following script to extract frames.
```shell
bash extract_frames.sh
```
If you only want to play with RGB frames (since extracting optical flow can be both time-comsuming and space-hogging), consider running the following script to extract **RGB-only** frames.
```shell
bash extract_rgb_frames.sh
```


### Generate filelist
Run the follow scripts to generate filelist in the format of videos and rawframes, respectively.
```shell
bash generate_video_filelist.sh
# execute the command below when rawframes are ready
bash generate_rawframes_filelist.sh
```

### Folder structure
In the context of the whole project (for kinetics400 only), the *minimal* folder structure will look like: (*minimal* means that some data are not necessary: for example, you may want to evaluate kinetics-400 using the original video format.)

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

For training and evaluating on Kinetics-400, please refer to [GETTING_STARTED.md](https://github.com/open-mmlab/mmaction/blob/master/GETTING_STARTED.md).