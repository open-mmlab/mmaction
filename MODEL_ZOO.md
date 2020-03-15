# Model Zoo

## Action Recognition

For action recognition, unless specified, models are trained on Kinetics-400. The version of Kinetics-400 we used contains 240436 training videos and 19796 testing videos. For TSN, we also train it on UCF-101, initialized with ImageNet pretrained weights. We also provide transfer learning results on UCF101 and HMDB51 for some algorithms. If you can not reproduce our testing results due to dataset unalignment, you can send us email to request our data. 

### TSN

#### Kinetics

| Modality | Pretrained | Backbone | Input | Top-1 | Top-5 |                                                              Download                                                                    |
| :------: | :--------: | :---------: | :--------: | :------------------------------------: | :------------------------------------: | -------------------------------------- |
|    RGB   |  ImageNet  | ResNet50 | 3seg  | 70.6  |  89.4  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/ucf101/tsn_2d_rgb_bninception_seg3_f1s1_b32_g8-98160339.pth)  |


#### UCF101

| Modality | Pretrained | Backbone | Input | Top-1 |                                                              Download                                                                    |
| :------: | :--------: | :---------: | :--------: | :--------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------: |
|    RGB   |  ImageNet  | BNInception | 3seg |  86.4    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/ucf101/tsn_2d_rgb_bninception_seg3_f1s1_b32_g8-98160339.pth)  |
|   TV-L1  |  ImageNet  | BNInception | 3seg |  87.7    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/ucf101/tsn_2d_flow_bninception_seg3_f1s1_b32_g8-151870b7.pth) |

### I3D

|  Modality  | Pretrained | Backbone | Input | Top-1 | Top-5 | Download |
| :--------: | :--------: | :------: | :---: | :---: | :---: | :------: |
|    RGB     |  ImageNet  |   I3D    | 64x1  | 71.1  | 89.3  |   ???    |
|    Flow    |  ImageNet  |   I3D    | 64x1  | 63.4  | 84.9  |   ???    |
| Two-Stream |  ImageNet  |   I3D    | 64x1  | 74.2  | 91.3  |    /     |

#### Transfer Learning

|  Modality  | Pretrained  | Backbone | Input | UCF101 | HMDB51 | Download (split1) |
| :--------: | :---------: | :------: | :---: | :----: | :----: | :---------------: |
|    RGB     | Kinetics400 |   I3D    | 64x1  |  71.1  |  89.3  |     ??? / ???     |
|    Flow    |  Kinetics   |   I3D    | 64x1  |  63.4  |  84.9  |     ??? / ???     |
| Two-Stream |  Kinetics   |   I3D    | 64x1  |  74.2  |  91.3  |         /         |

### SlowOnly

| Modality | Pretrained | Backbone | Input | Top-1 | Top-5 |                           Download                           |
| :------: | :--------: | :--------: | :--------: | :--------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| RGB  | None  | ResNet50 | 4x16 | 72.9  | 90.9  |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/slowonly_kinetics400_se_rgb_r50_seg1_4x16_scratch_epoch256-594abd88.pth) |
| RGB  | ImageNet | ResNet50 | 4x16 |  73.8  | 90.9  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/slowonly_kinetics400_se_rgb_r50_seg1_4x16_finetune_epoch150-46c79312.pth)  |
| RGB  | None  | ResNet50 | 8x8 | 74.8  | 91.9  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/slowonly_kinetics400_se_rgb_r50_seg1_8x8_scratch_epoch196-4aae9339.pth) |
| RGB  | ImageNet  | ResNet50 | 8x8 | 75.7  | 92.2  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/slowonly_kinetics400_se_rgb_r50_seg1_8x8_finetune_epoch150-519c2101.pth)  |
| RGB | None | ResNet101 | 8x8 | 76.5 | 92.7 | ??? |
| RGB | ImageNet | ResNet101 | 8x8 | 76.8 | 92.8 | ??? |

### SlowFast

| Modality | Pretrained | Backbone | Input | Top-1 | Top-5 | Download |
| :------: | :--------: | :------: | :---: | :---: | :---: | :------: |
|   RGB    |    None    | ResNet50 | 4x16  | 75.4  | 92.1  |   ???    |
|   RGB    |  ImageNet  | ResNet50 | 4x16  | 75.9  | 92.3  |   ???    |

### R(2+1)D
| Modality | Pretrained | Backbone | Input | Top-1 | Top-5 | Download |
| :------: | :--------: | :------: | :---: | :---: | :---: | :------: |
|   RGB    |    None    | ResNet34 |  8x8  | 63.7  | 85.9  |   ???    |
|   RGB    |   IG-65M   | ResNet34 |  8x8  | 74.4  | 91.7  |   ???    |
|   RGB    |    None    | ResNet34 | 32x2  | 71.8  | 90.4  |   ???    |
|   RGB    |   IG-65M   | ResNet34 | 32x2  | 80.3  | 94.7  |   ???    |

### CSN
| Modality | Pretrained | Backbone  | Input | Top-1 | Top-5 | Download |
| :------: | :--------: | :-------: | :---: | :---: | :---: | :------: |
|   RGB    |   IG-65M   | irCSN-152 | 32x2  | 82.6  | 95.7  |   ???    |
|   RGB    |   IG-65M   | ipCSN-152 | 32x2  | 82.7  | 95.6  |   ???    |

## Action Detection

For action detection, we release models trained on THUMOS14. 

### SSN

| Modality | Pretrained | mAP@0.10 | mAP@0.20 | mAP@0.30 | mAP@0.40 | mAP@0.50 |                           Download                           |
| :------: | :--------: | :------: | :------: | :------: | :------: | :------: | :----------------------------------------------------------: |
|   RGB    |  ImageNet  |  43.09%  |  37.95%  |  32.56%  |  25.71%  |  18.33%  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/thumos14/ssn_thumos14_rgb_bn_inception_tag-dac9ddb0.pth) |

## Spatial Temporal Action Detection

For spatial temporal action detection, we release

|   Model   | Pretrained | mAP@0.5 |                           Download                           |
| :-------: | :--------: | :-----: | :----------------------------------------------------------: |
| Fast-RCNN |  Kinetics  |  21.2   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/ava/fast_rcnn_ava2.1_nl_r50_c4_1x_f32s2_kin-e2495b48.pth) |
