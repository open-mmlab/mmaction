# Model Zoo

## Mirror sites
We use AWS as the main site to host our model zoo. Mirrors for aliyun will come soon.

## Datasets

### UCF-101 (split-1)
| Model | Modality | Pretrained | Data Preprocessing* | Avg. Acc. |                 Config                 |                                                              Download                                                                    |
| :---: | :------: | :--------: | :---------: | :--------: | :------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------: |
|  TSN  |    RGB   |  ImageNet  | 340x256 |  86.4%    | configs/ucf101/tsn_rgb_bninception.py  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/ucf101/tsn_2d_rgb_bninception_seg3_f1s1_b32_g8-98160339.pth)  |
|  TSN  |   TV-L1  |  ImageNet  | 340x256 |  87.7%    | configs/ucf101/tsn_flow_bninception.py | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/ucf101/tsn_2d_flow_bninception_seg3_f1s1_b32_g8-151870b7.pth) |


### Kinetics
| Model | Modality | Pretrained | Data Preprocessing* | Top-1 Acc. | Top-5 Acc. |                                     Config                                    |                                                                             Download                                                                         |
| :---: | :------: | :--------: | :--------: | :--------: | :--------: | :---------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  I3D  |   RGB    |  ImageNet  | 340x256 |  72.9%    |   90.8%    | configs/kinetics400/i3d_kinetics400_3d_rgb_r50_c3d_inflate3x1x1_seg1_f32s2.py | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/i3d_kinetics_rgb_r50_c3d_inflated3x1x1_seg1_f32s2_f32s2-b93cc877.pth) |
|  TSN  |   RGB    |  ImageNet  | 340x256 |  70.6%    |   89.4%    | configs/kinetics400/tsn_kinetics400_2d_rgb_r50_seg3_f1s1.py | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/tsn2d_kinetics400_rgb_r50_seg3_f1s1-b702e12f.pth) |
| SlowOnly 4x16 | RGB  | None  | SE256 | 72.9%  | 90.9%  | configs/kinetics400/slowonly_kinetics400_se_rgb_r50_seg1_4x16_scratch.py  |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/slowonly_kinetics400_se_rgb_r50_seg1_4x16_scratch_epoch256-594abd88.pth) |
| SlowOnly 4x16 | RGB  | ImageNet | SE256 |  73.8%  | 90.9%  | configs/kinetics400/slowonly_kinetics400_se_rgb_r50_seg1_4x16_finetune.py  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/slowonly_kinetics400_se_rgb_r50_seg1_4x16_finetune_epoch150-46c79312.pth)  |
| SlowOnly 8x8  | RGB  | None  | SE256 | 74.8%  | 91.9%  | configs/kinetics400/slowonly_kinetics400_se_rgb_r50_seg1_8x8_scratch.py  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/slowonly_kinetics400_se_rgb_r50_seg1_8x8_scratch_epoch196-4aae9339.pth) |
| SlowOnly 8x8  | RGB  | ImageNet  | SE256 | 75.7%  | 92.2%  | configs/kinetics400/slowonly_kinetics400_se_rgb_r50_seg1_8x8_finetune.py  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/slowonly_kinetics400_se_rgb_r50_seg1_8x8_finetune_epoch150-519c2101.pth)  |


PS: In data preprocessing, `340x256` denotes resizing all videos to 340x256,  `SE256` denotes rescaling short edge to 256 while keeping aspect ratio.

### THUMOS14
| Model | Modality | Pretrained | mAP@0.10 | mAP@0.20 | mAP@0.30 | mAP@0.40 | mAP@0.50 |                      Config                       |                                                               Download                                                              |
| :---: | :------: | :--------: | :------: | :------: | :------: | :------: | :------: | :-----------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------: |
|  SSN  |    RGB   |  ImageNet  |  43.09%  |  37.95%  |  32.56%  |  25.71%  |  18.33%  | configs/thumos14/ssn_thumos14_rgb_bn_inception.py | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/thumos14/ssn_thumos14_rgb_bn_inception_tag-dac9ddb0.pth) |

### AVA
|   Model   | Pretrained | mAP@0.5 |                              Config                               |                                                                Download                                                              |
| :-------: | :--------: | :-----: | :---------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: |
| Fast-RCNN |  Kinetics  |  21.2   | configs/ava/ava_fast_rcnn_nl_r50_c4_1x_kinetics_pretrain_crop.py  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/ava/fast_rcnn_ava2.1_nl_r50_c4_1x_f32s2_kin-e2495b48.pth) |
