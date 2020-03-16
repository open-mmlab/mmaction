# Model Zoo

## Action Recognition

For action recognition, unless specified, models are trained on Kinetics-400. The version of Kinetics-400 we used contains 240436 training videos and 19796 testing videos. For TSN, we also train it on UCF-101, initialized with ImageNet pretrained weights. We also provide transfer learning results on UCF101 and HMDB51 for some algorithms. Models with * are converted from other repos(including [VMZ](https://github.com/facebookresearch/VMZ) and [kinetics_i3d](https://github.com/deepmind/kinetics-i3d)), others are trained by ourselves. If you reproduce our testing results due to dataset unalignment, please submit a request at [get validation data](https://forms.gle/jmBiCDJButrLwpgc9).

### TSN

#### Kinetics

| Modality | Pretrained | Backbone | Input | Top-1 | Top-5 |                                                              Download                                                                    |
| :------: | :--------: | :---------: | :--------: | :------------------------------------: | :------------------------------------: | -------------------------------------- |
|    RGB   |  ImageNet  | ResNet50 | 3seg  | 70.6  |  89.4  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/tsn2d_kinetics400_rgb_r50_seg3_f1s1-b702e12f.pth)  |


#### UCF101

| Modality | Pretrained | Backbone | Input | Top-1 |                                                              Download                                                                    |
| :------: | :--------: | :---------: | :--------: | :--------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------: |
|    RGB   |  ImageNet  | BNInception | 3seg |  86.4    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/ucf101/tsn_2d_rgb_bninception_seg3_f1s1_b32_g8-98160339.pth)  |
|   TV-L1  |  ImageNet  | BNInception | 3seg |  87.7    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/ucf101/tsn_2d_flow_bninception_seg3_f1s1_b32_g8-151870b7.pth) |

### I3D

|  Modality  | Pretrained |   Backbone   | Input | Top-1 | Top-5 |                           Download                           |
| :--------: | :--------: | :----------: | :---: | :---: | :---: | :----------------------------------------------------------: |
|    RGB     |  ImageNet  | Inception-V1 | 64x1  | 71.1  | 89.3  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/i3d_kinetics400_se_rgb_inception_v1_seg1_f64s1_imagenet_deepmind-9b8e02b3.pth)* |
|    RGB     |  ImageNet  |   ResNet50   | 32x2  | 72.9  | 90.8  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/i3d_kinetics_rgb_r50_c3d_inflated3x1x1_seg1_f32s2_f32s2-b93cc877.pth) |
|    Flow    |  ImageNet  | Inception-V1 | 64x1  | 63.4  | 84.9  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/i3d_kinetics_flow_inception_v1_seg1_f64s1_imagenet_deepmind-92059771.pth)* |
| Two-Stream |  ImageNet  | Inception-V1 | 64x1  | 74.2  | 91.3  |                              /                               |

### SlowOnly

| Modality | Pretrained | Backbone | Input | Top-1 | Top-5 |                           Download                           |
| :------: | :--------: | :--------: | :--------: | :--------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| RGB  | None  | ResNet50 | 4x16 | 72.9  | 90.9  |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/slowonly_kinetics400_se_rgb_r50_seg1_4x16_scratch_epoch256-594abd88.pth) |
| RGB  | ImageNet | ResNet50 | 4x16 |  73.8  | 90.9  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/slowonly_kinetics400_se_rgb_r50_seg1_4x16_finetune_epoch150-46c79312.pth)  |
| RGB  | None  | ResNet50 | 8x8 | 74.8  | 91.9  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/slowonly_kinetics400_se_rgb_r50_seg1_8x8_scratch_epoch196-4aae9339.pth) |
| RGB  | ImageNet  | ResNet50 | 8x8 | 75.7  | 92.2  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/slowonly_kinetics400_se_rgb_r50_seg1_8x8_finetune_epoch150-519c2101.pth)  |
| RGB | None | ResNet101 | 8x8 | 76.5 | 92.7 | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/slowonly_kinetics400_se_rgb_r101_8x8_scratch-8de47237.pth) |
| RGB | ImageNet | ResNet101 | 8x8 | 76.8 | 92.8 | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/slowonly_kinetics400_se_rgb_r101_8x8_finetune-b8455f97.pth) |

### SlowFast

| Modality | Pretrained | Backbone | Input | Top-1 | Top-5 |                           Download                           |
| :------: | :--------: | :------: | :---: | :---: | :---: | :----------------------------------------------------------: |
|   RGB    |    None    | ResNet50 | 4x16  | 75.4  | 92.1  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/slowfast_kinetics400_se_rgb_r50_4x16_scratch-2448c56c.pth) |
|   RGB    |  ImageNet  | ResNet50 | 4x16  | 75.9  | 92.3  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/slowfast_kinetics400_se_rgb_r50_4x16_finetune-4623cf03.pth) |

### R(2+1)D
| Modality | Pretrained | Backbone | Input | Top-1 | Top-5 |                           Download                           |
| :------: | :--------: | :------: | :---: | :---: | :---: | :----------------------------------------------------------: |
|   RGB    |    None    | ResNet34 |  8x8  | 63.7  | 85.9  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/r2plus1d_kinetics400_se_rgb_r34_f8s8_scratch-1f576444.pth) |
|   RGB    |   IG-65M   | ResNet34 |  8x8  | 74.4  | 91.7  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/r2plus1d_kinetics400_se_rgb_r34_f8s8_finetune-c3abbbfc.pth) |
|   RGB    |    None    | ResNet34 | 32x2  | 71.8  | 90.4  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/r2plus1d_kinetics400_se_rgb_r34_f32s2_scratch-97f56158.pth) |
|   RGB    |   IG-65M   | ResNet34 | 32x2  | 80.3  | 94.7  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/r2plus1d_kinetics400_se_rgb_r34_f32s2_finetune-9baa39ea.pth) |

### CSN
| Modality | Pretrained | Backbone  | Input | Top-1 | Top-5 |                           Download                           |
| :------: | :--------: | :-------: | :---: | :---: | :---: | :----------------------------------------------------------: |
|   RGB    |   IG-65M   | irCSN-152 | 32x2  | 82.6  | 95.7  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/ircsn_kinetics400_se_rgb_r152_f32s2_ig65m_fbai-9d6ed879.pth)* |
|   RGB    |   IG-65M   | ipCSN-152 | 32x2  | 82.7  | 95.6  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/ipcsn_kinetics400_se_rgb_r152_f32s2_ig65m_fbai-ef39b9e3.pth)* |

### Transfer Learning

| Model | Modality  | Pretrained | Backbone | Input | UCF101 | HMDB51 |                      Download (split1)                       |
| ----- | :-------: | :--------: | :------: | :---: | :----: | :----: | :----------------------------------------------------------: |
| I3D   |    RGB    |  Kinetics  |   I3D    | 64x1  |  94.8  |  72.6  | [UCF101](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/ucf101/i3d_ucf101_split1_rgb_f64s1_kinetics400ft-36201298.pth) / [HMDB51](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/ucf101/i3d_hmdb51_split1_rgb_f64s1_kinetics400ft-1ffcf11f.pth) |
| I3D   |   Flow    |  Kinetics  |   I3D    | 64x1  |  96.6  |  79.2  | [UCF101](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/ucf101/i3d_ucf101_split1_flow_f64s1_kinetics400ft-93ed9ecd.pth) / [HMDB51](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/ucf101/i3d_hmdb51_split1_flow_f64s1_kinetics400ft-2981c797.pth) |
| I3D   | TwoStream |  Kinetics  |   I3D    | 64x1  |  97.8  |  80.8  |                              /                               |

## Action Detection

For action detection, we release models trained on THUMOS14.

### SSN

| Modality | Pretrained |  Backbone   | mAP@0.10 | mAP@0.20 | mAP@0.30 | mAP@0.40 | mAP@0.50 |                           Download                           |
| :------: | :--------: | :---------: | :------: | :------: | :------: | :------: | :------: | :----------------------------------------------------------: |
|   RGB    |  ImageNet  | BNInception |  43.09%  |  37.95%  |  32.56%  |  25.71%  |  18.33%  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/thumos14/ssn_thumos14_rgb_bn_inception_tag-dac9ddb0.pth) |

## Spatial Temporal Action Detection

For spatial temporal action detection, we release models trained on AVA.

| Modality |   Model   | Pretrained |  Backbone  | mAP@0.5 |                           Download                           |
| :------: | :-------: | :--------: | :--------: | :-----: | :----------------------------------------------------------: |
|   RGB    | Fast-RCNN |  Kinetics  | NL-I3D R50 |  21.2   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/ava/fast_rcnn_ava2.1_nl_r50_c4_1x_f32s2_kin-e2495b48.pth) |
