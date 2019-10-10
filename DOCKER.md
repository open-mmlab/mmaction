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

When run the container, Please follow step [GETTING_STARTED.md](https://github.com/open-mmlab/mmaction/blob/master/GETTING_STARTED.md) to use mmaction.