# using Docker to set environment of mmaction

## Requirements

We've been testing/build from ubuntu 18.04 LTS & docker version 19.03.1 (with docker API version 1.40). If you want to building docker images, you could have

- Docker Engine
- nvidia-docker (to start container with GPUs)
- Disk space (a lot)

## Install Docker Engine (Ubuntu version)

```
$ curl -fsSL https://get.docker.com -o get-docker.sh
$ sh get-docker.sh 
```

## Install Nvidia-Docker

You could update from [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

## Build the images

You could see the ```Dockerfile``` on [this](https://github.com/open-mmlab/mmaction) repository. So you can copy this file and build as manually or clone this repository.

```
$ git clone --recursive https://github.com/open-mmlab/mmaction
$ cd mmaction
$ docker build -t mmaction .
```

So when you building this image. The image will not successfully because we want to modified in this code. So you can clone repository in container manually from next step below.

## Run container from images

```
$ docker run --name mmaction --gpus all -it -v /path/to/your/data:/root mmaction
```

When you run this command. You'll see the container that have opencv 4.1.0 (with CUDA). ***Don't forget to use GPUs attach to the container*** You can follow my instruction to install the [mmaction](https://github.com/open-mmlab/mmaction)

So the step that we could do follow by: ```clone mmaction -> install decord -> dense_flow -> mmcv -> mmaction```

### 1.) clone this repository into container
```
# inside container
# clone this repository
$ git clone --recursive https://github.com/open-mmlab/mmaction.git
```

### 2.) install decord
```
$ cd third_party/decord \
    && mkdir build \
    && cd build \
    && cmake .. -DUSE_CUDA=0 \
    && make \
    && cd ../python \
    && python setup.py install --user
```

### 3.) install dense_flow
```
$ cd third_party/dense_flow \
    && mkdir build \
    && cd build \
    && OpenCV_DIR=/opencv-4.1.0/build cmake .. \
    && make -j
```
### 4.) install mmcv
```
$ git clone --recursive https://github.com/open-mmlab/mmcv.git \
    && cd mmcv \
    && pip install -e .
```

### 5.) install mmaction
```
$ cd mmaction \ 
    && chmod 777 compile.sh \
    && ./compile.sh \
    && python3 setup.py develop
```

When finished installation, Please follow step [GETTING_STARTED.md](https://github.com/open-mmlab/mmaction/blob/master/GETTING_STARTED.md) to use mmaction.