## Preparing HMDB51

For more details, please refer to the official [website](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/). We provide scripts with documentations. Before we start, please make sure that the directory is located at `$MMACTION/data_tools/hmdb51/`.

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
Before extraction, please refer to `DATASET.md` for installing [dense_flow](https://github.com/yjxiong/dense_flow).
If you have some SSD, then we recommend extracting frames there for better I/O performance. The extracted frames (RGB + Flow) will take up ~24GB.
```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/hmdb51_extracted/
ln -s /mnt/SSD/hmdb51_extracted/ ../data/hmdb51/rawframes
```

If you didn't install dense_flow in the installation or only want to play with RGB frames (since extracting optical flow can be both time-comsuming and space-hogging), consider running the following script to extract **RGB-only** frames.
```shell
bash extract_rgb_frames.sh
```

If both rgb and optical flow are required, run the following script to extract frames alternatively.
```shell
bash extract_frames.sh
```

### Generate filelist
Run the follow script to generate filelist in the format of rawframes and videos.
```shell
bash generate_filelist.sh
```

### Folder structure
In the context of the whole project (for ucf101 only), the folder structure will look like:
```
mmaction
├── mmaction
├── tools
├── configs
├── data
│   ├── hmdb51
│   │   ├── hmdb51_{train,val}_split_{1,2,3}_rawframes.txt
│   │   ├── hmdb51_{train,val}_split_{1,2,3}_videos.txt
│   │   ├── annotations
│   │   ├── videos
│   │   │   ├── brush_hair
│   │   │   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0.avi

│   │   │   ├── wave
│   │   │   │   ├── 20060723sfjffbartsinger_wave_f_cm_np1_ba_med_0.avi
│   │   ├── rawframes
│   │   │   ├── brush_hair
│   │   │   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0
│   │   │   │   │   ├── img_00001.jpg
│   │   │   │   │   ├── img_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_x_00001.jpg
│   │   │   │   │   ├── flow_x_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_y_00001.jpg
│   │   │   │   │   ├── flow_y_00002.jpg
│   │   │   ├── ...
│   │   │   ├── wave
│   │   │   │   ├── 20060723sfjffbartsinger_wave_f_cm_np1_ba_med_0
│   │   │   │   ├── ...
│   │   │   │   ├── winKen_wave_u_cm_np1_ri_bad_1

```

For training and evaluating on HMDB51, please refer to [GETTING_STARTED.md](https://github.com/open-mmlab/mmaction/blob/master/GETTING_STARTED.md).