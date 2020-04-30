FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

MAINTAINER @mynameismaxz (github.com/mynameismaxz)

# install all-of-package
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:jonathonf/ffmpeg-4 -y && \
    apt-get update && \
    apt-get install -y build-essential \
    python-pip \
    python-dev \
    python-numpy \
    python3-dev \
    python3-setuptools \
    python3-numpy \
    python3-pip \
    make \
    cmake \
    libavcodec-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    ffmpeg \
    wget \
    git \
    libcurl4-gnutls-dev \
    zlib1g-dev \
    liblapack-dev \
    libatlas-base-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libswscale-dev \
    libdc1394-22-dev \
    libzip-dev \
    libboost* \
    zip \
    unrar \
    yasm \
    pkg-config \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    libavformat-dev \
    libpq-dev \
	libxine2-dev \
	libglew-dev \
	libtiff5-dev \
	zlib1g-dev \
	libjpeg-dev \
	libpng12-dev \
	libjasper-dev \
	libavcodec-dev \
	libavformat-dev \
	libavutil-dev \
	libpostproc-dev \
	libswscale-dev \
	libeigen3-dev \
	libtbb-dev \
	libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /data

WORKDIR /data

# unlink old-python (python2) & make new symbolic-link for python3
RUN unlink /usr/bin/python \
    && unlink /usr/bin/pip \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip \
    && pip install --upgrade pip

# install essential python package
RUN pip install torchvision==0.4.0 \
    cython==0.29.11 \
    numpy==1.16.4 \
    scipy \
    pandas \
    matplotlib \
    scikit-learn

# 1 st step - clone repository & install opencv 4.1.0 (using a lot of time!!)
RUN wget -O OpenCV-4.1.0.zip https://github.com/opencv/opencv/archive/4.1.0.zip \
    && unzip OpenCV-4.1.0.zip \
    && rm -rf OpenCV-4.1.0.zip \
    && wget -O OpenCV_contrib-4.1.0.zip https://github.com/opencv/opencv_contrib/archive/4.1.0.zip \
    && unzip OpenCV_contrib-4.1.0.zip \
    && rm -rf OpenCV_contrib-4.1.0.zip \
    && cd opencv-4.1.0 \
    && mkdir build \
    && cd build \
    ### using cmake refer from INSTALLATION.md default file ###
    && cmake \ 
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_CUDA=ON \
        -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.1.0/modules/ \
        -DWITH_TBB=ON \
        -DBUILD_opencv_cnn_3dobj=OFF \
        -DBUILD_opencv_dnn=OFF \
        -DBUILD_opencv_dnn_modern=OFF \
        -DBUILD_opencv_dnns_easily_fooled=OFF \
        -DOPENCV_ENABLE_NONFREE=ON \
        .. \
    && make -j

# clone repository (mmaction)
RUN git clone --recursive https://github.com/open-mmlab/mmaction.git

# install cmake first
RUN wget --no-check-certificate https://cmake.org/files/v3.9/cmake-3.9.0.tar.gz \
    && tar -zxvf cmake-3.9.0.tar.gz \
    && rm -rf cmake-3.9.0.tar.gz \
    && cd cmake-3.9.0 \
    && ./bootstrap --system-curl \
    && make -j && make install

# install decord
RUN cd mmaction/third_party/decord \
    && mkdir -p build \
    && cd build \
    && cmake .. -DUSE_CUDA=0 \
    && make -j \
    && cd ../python \
    && python3 setup.py install --user

# install dense flow
RUN cd mmaction/third_party/dense_flow \
    && mkdir build \
    && cd build \
    && OpenCV_DIR=/data/opencv-4.1.0/build/ cmake .. \
    && make -j

# install mmcv
RUN pip install mmcv==0.2.16

# setup mmaction
RUN cd mmaction \ 
    && chmod 777 compile.sh \
    && ./compile.sh \
    && python3 setup.py develop
