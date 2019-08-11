# MMAction

## Introduction
MMAction is an open source toolbox for action understanding based on PyTorch.
It is a part of the [open-mmlab](https://github.com/open-mmlab) project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

### Major Features
- MMAction is capable of dealing with all of the tasks below.

  - action recognition from trimmed videos
  - temporal action detection (also known as action localization) in untrimmed videos
  - spatial-temporal action detection in untrimmed videos.


- Support for various datasets

  Video datasets have emerging throughout the recent years and have greatly fostered the development of this field.
  MMAction provides tools to deal with various datasets.

- Support for multiple action understanding frameworks

  MMAction implements popular frameworks such as TSN, I3D for action recognition and SSN for temporal action detection.
  For the recently introduced spatial temporal atomic action detection, a Fast-RCNN baseline is provided.

- Modular design

  The tasks in human action understanding share some common aspects such as backbones, and long-term and short-term sampling schemes.
  Also, tasks can benefit from each other. For example, a better backbone for action recognition will bring performance gain for action detection.
  Modular design enables us to view action understanding in a more integrated perspective.

## License
The project is release under the [Apache 2.0 license](https://github.com/open-mmlab/mmaction/blob/master/LICENSE).

## Updates

v0.1.0 (19/06/2019)
- MMAction is online!

## Model zoo
Results and reference models are available in the [model zoo](https://github.com/open-mmlab/mmaction/blob/master/MODEL_ZOO.md).

## Installation
Please refer to [INSTALL.md](https://github.com/open-mmlab/mmaction/blob/master/INSTALL.md) for installation.

## Data preparation
Please refer to [DATASET.md](https://github.com/open-mmlab/mmaction/blob/master/DATASET.md) for a general knowledge of data preparation.
Detailed documents for the supported datasets are available in `data_tools/`.

## Get started
Please refer to [GETTING_STARTED.md](https://github.com/open-mmlab/mmaction/blob/master/GETTING_STARTED.md) for detailed examples.

## Contributing
We appreciate all contributions to improve MMAction.
Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmaction/blob/master/CONTRIBUTING.md) for the contributing guideline.

## Citation
If you use our codebase or models in your research, please cite this work.
We will release a technical report later.
```
@misc{mmaction2019,
  author =       {Yue Zhao, Yuanjun Xiong, Dahua Lin},
  title =        {MMAction},
  howpublished = {\url{https://github.com/open-mmlab/mmaction}},
  year =         {2019}
}
```

## Contact
If you have any question, please file an issue or contact the author:
```
Yue Zhao: thuzhaoyue@gmail.com
```
