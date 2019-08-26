FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

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
	libgtk2.0-dev

WORKDIR /data

RUN ls -la /usr/local/cuda-9.0/lib64

# install opencv 4.1.0
RUN wget -O OpenCV-4.1.0.zip https://github.com/opencv/opencv/archive/4.1.0.zip \
    && unzip OpenCV-4.1.0.zip \
    && wget -O OpenCV_contrib-4.1.0.zip https://github.com/opencv/opencv_contrib/archive/4.1.0.zip \
    && unzip OpenCV_contrib-4.1.0.zip \
    && cd opencv-4.1.0 \
    && mkdir build \
    && cd build \
    && cmake -DBUILD_TIFF=ON \
       -DBUILD_opencv_java=OFF \
       -DBUILD_SHARED_LIBS=OFF \
       -DWITH_CUDA=ON \
       -DENABLE_FAST_MATH=1 \
       -DCUDA_FAST_MATH=1 \
       -DWITH_CUBLAS=1 \
       -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0/lib64 \
       -DCMAKE_INSTALL_PREFIX=/opencv \
       ##
       ## Should compile for most card
       ## 3.5 binary code for devices with compute capability 3.5 and 3.7,
       ## 5.0 binary code for devices with compute capability 5.0 and 5.2,
       ## 6.0 binary code for devices with compute capability 6.0 and 6.1,
       -DCUDA_ARCH_BIN='3.0 3.5 3.7 5.0 5.2 6.0 6.1' \
       -DCUDA_ARCH_PTX="" \
       ##
       ## AVX in dispatch because not all machines have it
       -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.1.0/modules \
       -DCPU_DISPATCH=AVX,AVX2 \
       -DENABLE_PRECOMPILED_HEADERS=OFF \
       -DWITH_OPENGL=OFF \
       -DWITH_OPENCL=OFF \
       -DWITH_QT=OFF \
       -DWITH_IPP=ON \
       -DWITH_TBB=ON \
       -DFORCE_VTK=ON \
       -DWITH_EIGEN=ON \
       -DWITH_V4L=ON \
       -DWITH_XINE=ON \
       -DWITH_GDAL=ON \
       -DWITH_1394=OFF \
       -DWITH_FFMPEG=OFF \
       -DBUILD_PROTOBUF=OFF \
       -DBUILD_TESTS=OFF \
       -DBUILD_PERF_TESTS=OFF \
       -DCMAKE_BUILD_TYPE=RELEASE \
       -DWITH_GTK=ON \
        .. \
    # && cmake -DCMAKE_BUILD_TYPE=Release \
    #     -DWITH_CUDA=ON \
    #     -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 \
    #     -DCUDA_ARCH_BIN='3.0 3.5 5.0 6.0 6.2' \
    #     -DCUDA_ARCH_PTX="" \
    #     -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.1.0/modules/ \
    #     -DWITH_TBB=ON \
    #     -DBUILD_opencv_cnn_3dobj=OFF \
    #     -DBUILD_opencv_dnn=OFF \
    #     -DBUILD_opencv_dnn_modern=OFF \
    #     -DBUILD_opencv_dnns_easily_fooled=OFF \
    #     -DOPENCV_ENABLE_NONFREE=ON .. \
    && make -j

# install cmake first
RUN wget --no-check-certificate https://cmake.org/files/v3.9/cmake-3.9.0.tar.gz \
    && tar -zxvf cmake-3.9.0.tar.gz \
    && cd cmake-3.9.0 \
    && ./bootstrap --system-curl \
    && make -j && make install

# install decord
RUN git clone --recursive https://github.com/open-mmlab/mmaction.git \
    && cd mmaction/third_party/decord \
    && mkdir -p build \
    && cd build \
    && cmake .. -DUSE_CUDA=0 \
    && make -j4 \
    && cd ../python \
    && python3 setup.py install --user

# install dense flow
RUN cd mmaction/third_party/dense_flow \
    && mkdir build \
    && cd build \
    # && find / -name cudaarithm.hpp \
    && OpenCV_DIR=/data/opencv-4.1.0/build/ cmake .. \
    # && cmake .. \
    && make -j

RUN pip3 install http://download.pytorch.org/whl/cu90/torch-1.0.1-cp35-cp35m-linux_x86_64.whl

RUN git clone --recursive https://github.com/open-mmlab/mmcv.git \
    && cd mmcv \
    && pip3 install -e .

RUN unlink /usr/bin/python \
    && unlink /usr/bin/pip

RUN ln -s /usr/bin/python3 /usr/bin/python & \
    ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip

RUN pip install cython>=0.29.11 numpy==1.16.4 scipy pandas matplotlib scikit-learn \
    && cd mmaction \ 
    && chmod 777 compile.sh \
    && CUDA_VISIBLE_DEVICES=0 ./compile.sh \
    && python3 setup.py develop