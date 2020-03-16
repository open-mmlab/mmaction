# Installation

```shell
git clone --recursive https://github.com/open-mmlab/mmaction.git
```

## Requirements

- Linux
- Python 3.5+
- PyTorch 1.0+
- CUDA 9.0+
- NVCC 2+
- GCC 4.9+
- ffmpeg 4.0+
- [mmcv](https://github.com/open-mmlab/mmcv).
  Note that you are strongly recommended to clone the master branch and build from scratch since some of the features have not been added in the latest release.
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
sudo apt-get install -y software-properties-common
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

Note that Dense_flow now supports OpenCV 4.1.0, 3.1.0 and 2.4.13.
The master branch is for 4.1.0. For those with 2.4.13, please refer to the lines with strikethrough.

<del>
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
</del>

(a) Install OpenCV=4.1.0

0. (For CUDA 10.0 only) CUDA 9.x should have no problem.
  Video decoder is deprecated in CUDA 10.0.
To handle this, download [NVIDIA VIDEO CODEC SDK](https://developer.nvidia.com/nvidia-video-codec-sdk) and copy the header files to your cuda path (`/usr/local/cuda-10.0/include/` for example).
Note that you may have to do as root.

```shell
unzip Video_Codec_SDK_9.0.20.zip
cp Video_Codec_SDK_9.0.20/include/nvcuvid.h /usr/local/cuda-10.0/include/
cp Video_Codec_SDK_9.0.20/include/cuviddec.h /usr/local/cuda-10.0/include/
cp Video_Codec_SDK_9.0.20/Lib/linux/stubs/x86_64/libnvcuvid.so /usr/local/cuda-10.0/lib64/libnvcuvid.so.1
```

1. Obtain required packages for building OpenCV 4.1.0 (duplicated with requirements for Decord in part)

```shell
sudo apt-get install -y liblapack-dev libatlas-base-dev
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt update
sudo apt install -y ffmpeg
```

2. Obtain OpenCV 4.1.0 and its extra modules (optflow, etc.) by

```shell
cd third_party
wget -O OpenCV-4.1.0.zip wget https://github.com/opencv/opencv/archive/4.1.0.zip
unzip OpenCV-4.1.0.zip
wget -O OpenCV_contrib-4.1.0.zip https://github.com/opencv/opencv_contrib/archive/4.1.0.zip
unzip OpenCV_contrib-4.1.0.zip
```

3. Build OpenCV 4.1.0 from scratch (due to some custom settings)

```
cd opencv-4.1.0
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.1.0/modules/ -DWITH_TBB=ON -DBUILD_opencv_cnn_3dobj=OFF -DBUILD_opencv_dnn=OFF -DBUILD_opencv_dnn_modern=OFF -DBUILD_opencv_dnns_easily_fooled=OFF -DOPENCV_ENABLE_NONFREE=ON ..
make -j
```

Note that `-DOPENCV_ENABLE_NONFREE=ON` is explicitly set to enable *warped flow* proposed in [TSN](https://arxiv.org/abs/1608.00859).
You can skip this argument to speed up the compilation if you do not intend to use it.

(b) Build dense_flow
```shell
cd third_party/dense_flow
# dense_flow dependencies
sudo apt-get -qq install libzip-dev libboost-all-dev
mkdir build && cd build
# deprecated:
# OpenCV_DIR=../../opencv-2.4.13/build cmake ..
OpenCV_DIR=../../opencv-4.1.0/build cmake ..
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

