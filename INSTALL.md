# Installation

```shell
git clone --recursive https://github.com/open-mmlab/mmaction.git
```

## Requirements

- Linux
- Python 3.5+
- PyTorch 1.0+
- CUDA 9.0+
- NCCV 2+
- GCC 4.9+
- ffmpeg 4.0+
- [mmcv](https://github.com/open-mmlab/mmcv)
- [Decord](https://github.com/zhreshold/decord)
- [dense_flow](https://github.com/yjxiong/dense_flow)

Note that the last two will be contained in this codebase as a submodule.

## Install Pre-requisites
### Install Decord (Optional)
[Decord](https://github.com/zhreshold/decord) is an efficient video loader with smart shuffling.
This is required when you want to use videos as the input format for training and is more efficient than OpenCV, useds by [mmcv](https://github.com/open-mmlab/mmcv).
If you just want to have a quick experience with MMAction, you can simply skip this step.
The installation steps follow decord's documentation.

(a) install the required packages by running:

```shell
# official PPA comes with ffmpeg 2.8, which lacks tons of features, we use ffmpeg 4.0 here
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-setuptools make cmake 
libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev
# note: make sure you have cmake 3.8 or later, you can install from cmake official website if it's too old
sudo apt-get install ffmpeg
```

(b) Build the library from source

```shell
cd third_party/decord
mkdir build && cd build
cmake .. -DUSE_CUDA=0
make
```

(c) Install python binding

```shell
cd ../python
python setup.py install --user
cd ../../
```

### Install dense_flow (Optional)
[Dense_flow](https://github.com/yjxiong/dense_flow) is used to calculate the optical flow of videos.
If you just want to have a quick experience with MMAction without taking pain of installing opencv, you can skip this step.

(a) Install OpenCV=2.4.13
```shell
cd third_party/
sudo apt-get -qq install libopencv-dev build-essential checkinstall cmake pkg-config yasm libjpeg-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libv4l-dev python-dev python-numpy libtbb-dev libqt4-dev libgtk2.0-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils

wget -O OpenCV-2.4.13.zip https://github.com/Itseez/opencv/archive/2.4.13.zip
unzip OpenCV-2.4.13.zip

cd opencv-2.4.13
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_TBB=ON -D WITH_V4L=ON ..
make -j32
cd ../../../
```

(b) Build dense_flow
```shell
cd third_party/dense_flow
# dense_flow dependencies
sudo apt-get -qq install libzip-dev
mkdir build && cd build
OpenCV_DIR=../../third_party/opencv-2.4.13/build cmake .. -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF
make -j
```

## Install MMAction
(a) Install Cython
```shell
pip install cython
```
(b) Compile CUDA extensions
```shell
./compile.sh
```
(c) Install mmaction
```shell
python setup.py develop
```

Please refer to [DATASET.md](https://github.com/open-mmlab/mmaction/blob/master/DATASET.md) to get familar with the data preparation and to [GETTING_STARTED.md](https://github.com/open-mmlab/mmaction/blob/master/GETTING_STARTED.md) to use MMAction.

