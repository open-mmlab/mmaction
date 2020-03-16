# Getting Started

This document provides basic tutorials for the usage of MMAction.
For installation, please refer to [INSTALL.md](https://github.com/open-mmlab/mmaction/blob/master/INSTALL.md).
For data deployment, please refer to [DATASET.md](https://github.com/open-mmlab/mmaction/blob/master/DATASET.md).


## An example on UCF-101
We first give an example of testing and training action recognition models on UCF101.
### Prepare data
First of all, please follow [PREPARING_UCF101.md](https://github.com/open-mmlab/mmaction/blob/master/data_tools/ucf101/PREPARING_UCF101.md) for data preparation.

### Test a reference model
Reference models are stored in [MODEL_ZOO.md](https://github.com/open-mmlab/mmaction/blob/master/MODEL_ZOO.md).
We download a reference model spatial stream BN-Inception at `$MMACTION/modelzoo` using:
```shell
wget -c https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/ucf101/tsn_2d_rgb_bninception_seg3_f1s1_b32_g8-98160339.pth -P ./modelzoo/
```
Then, together with provided configs files, we run the following code to test with multiple GPUs:
```shell
./tools/dist_test_recognizer.sh test_configs/TSN/ucf101/tsn_rgb_bninception.py tsn_2d_rgb_bninception_seg3_f1s1_b32_g8-98160339.pth 8
```

When testing 3D ConvNets, the oversample we used is 10 clips x 3 crops by default. For some extremely large models, it might be difficult for so many samples to fit on 1 GPU. When it happens, you can use commands for heavy test instead:
```shell
./tools/dist_test_recognizer_heavy.sh test_configs/CSN\ircsn_kinetics400_se_rgb_r152_seg1_32x2.py ircsn_kinetics400_se_rgb_r152_f32s2_ig65m_fbai-9d6ed879.pth 8 --batch_size=5
```


### Train a model with multiple GPUs

To reproduce the model, we provide training scripts as follows:
```shell
./tools/dist_train_recognizer.sh configs/TSN/ucf101/tsn_rgb_bninception.py 8 --validate
```
- `--validate`: performs evaluation every k (default=1) epochs during the training, which help diagnose training process.


## More examples
The procedure is not limited to action recognition in UCF101.
To perform spatial-temporal detection on AVA, we can train a baseline model by running
```shell
./tools/dist_train_detector.sh configs/ava/ava_fast_rcnn_nl_r50_c4_1x_kinetics_pretrain_crop.py 8 --validate
```
and evaluate a reference model by running
```shell
wget -c wget -c https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/ava/fast_rcnn_ava2.1_nl_r50_c4_1x_f32s2_kin-e2495b48.pth -P modelzoo/
python tools/test_detector.py ava_fast_rcnn_nl_r50_c4_1x_kinetics_pretrain_crop.py modelzoo/fast_rcnn_ava2.1_nl_r50_c4_1x_f32s2_kin-e2495b48.pth --out ava_fast_rcnn_nl_r50_multiscale.pkl --gpus 8 --eval bbox
```

To perform temporal action detection on THUMOS14, we can training a baseline model by running
```shell
./tools/dist_train_localizer.sh configs/thumos14/ssn_thumos14_rgb_bn_inception.py 8
```
and evaluate a reference model by running
```shell
wget -c wget -c https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/thumos14/ssn_thumos14_rgb_bn_inception_tag-dac9ddb0.pth -P modelzoo/
python tools/test_detector.py configs/thumos14/ssn_thumos14_rgb_bn_inception.py modelzoo/ssn_thumos14_rgb_bn_inception_tag-dac9ddb0.pth --gpus 8 --out ssn_thumos14_rgb_bn_inception.pkl --eval thumos14
```


## More Abstract Usage

## Inference with pretrained models
We provide testing scripts to evaluate a whole dataset.

### Test a dataset
```shell
python tools/test_${ARCH}.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [other task-specific arguments]
```
Arguments:
- `${ARCH}` could be
    - "recognizer" for action recognition (TSN, I3D, SlowFast, R(2+1)D, CSN, ...)
    - "localizer" for temporal action detection/localization (SSN)
    - "detector" for spatial-temporal action detection (a re-implmented Fast-RCNN baseline)
- `${CONFIG_FILE}` is the config file stored in `$MMACTION/test_configs`.
- `${CHECKPOINT_FILE}` is the checkpoint file.
    Please refer to [MODEL_ZOO.md](https://github.com/open-mmlab/mmaction/blob/master/MODEL_ZOO.md) for more details.


## Train a model
MMAction implements distributed training and non-distributed training, powered by the same engine of [mmdetection](https://github.com/open-mmlab/mmdetection).


### Train with multiple GPUs (Recommended)
Training with multiple GPUs follows the rules below:

```shell
./tools/dist_train_${ARCH}.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```
- ${ARCH} could be
    - "recognizer" for action recognition (TSN, I3D, ...)
    - "localizer" for temporal action detection/localization (SSN)
    - "detector" for spatial-temporal action detection (a re-implmented Fast-RCNN baseline)
- ${CONFIG_FILE} is the config file stored in `$MMACTION/configs`.
- ${GPU_NUM} is the number of GPU (default: 8). If you are using number other than 8, please adjust the learning rate in the config file linearly.
