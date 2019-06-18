## Preparing THUMOS-14

For more details, please refer to the [official website](https://www.crcv.ucf.edu/THUMOS14/download.html). We provide scripts with documentations. Before we start, please make sure that the directory is located at `$MMACTION/data_tools/thumos14/`.

### Prepare annotations
First of all, run the following script to prepare annotations.
```shell
bash download_annotations.sh
```

### Prepare videos
Then, use the following script to prepare videos. 
```shell
bash download_videos.sh
```

### Extract frames
Now it is time to extract frames from videos. 
Before extraction, please refer to `DATASET.md` for installing [dense_flow].
If you have some SSD, then we strongly recommend extracting frames there for better I/O performance. 
```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/thumos14_extracted/
ln -s /mnt/SSD/thumos14_extracted/ ../data/thumos14/rawframes/
```
Afterwards, run the following script to extract frames.
```shell
bash extract_frames.sh
```

### Fetching proposal files
Run the follow scripts to fetch pre-computed tag proposals.
```shell
bash fetch_tag_proposals.sh
```

### Folder structure
In the context of the whole project (for thumos14 only), the folder structure will look like: 

```
mmaction
├── mmaction
├── tools
├── configs
├── data
│   ├── thumos14
│   │   ├── thumos14_tag_val_normalized_proposal_list.txt
│   │   ├── thumos14_tag_test_normalized_proposal_list.txt
│   │   ├── annotations
│   │   ├── videos_val
│   │   │   ├── video_validation_0000001.mp4
│   │   │   ├── ...
│   │   ├── videos_test
│   │   │   ├── video_test_0000001.mp4
│   │   ├── rawframes
│   │   │   ├── video_validation_0000001
|   │   │   │   ├── img_00001.jpg
|   │   │   │   ├── img_00002.jpg
|   │   │   │   ├── ...
|   │   │   │   ├── flow_x_00001.jpg
|   │   │   │   ├── flow_x_00002.jpg
|   │   │   │   ├── ...
|   │   │   │   ├── flow_y_00001.jpg
|   │   │   │   ├── flow_y_00002.jpg
|   │   │   │   ├── ...
│   │   │   ├── ...
│   │   │   ├── video_test_0000001
```

For training and evaluating on THUMOS-14, please refer to [GETTING_STARTED.md](https://github.com/open-mmlab/mmaction/blob/master/GETTING_STARTED.md).