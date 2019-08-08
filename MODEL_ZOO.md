# Model Zoo

## Mirror sites
We use AWS as the main site to host our model zoo. Mirrors for aliyun will come soon.

## Datasets

### UCF-101 (split-1)
| Model | Modality | Pretrained |  Avg. Acc. |                 Config                 |                                                              Download                                                                    |
| :---: | :------: | :--------: | :--------: | :------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------: |
|  TSN  |    RGB   |  ImageNet  |   86.4%    | configs/ucf101/tsn_rgb_bninception.py  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/ucf101/tsn_2d_rgb_bninception_seg3_f1s1_b32_g8-98160339.pth)  |
|  TSN  |   TV-L1  |  ImageNet  |   87.7%    | configs/ucf101/tsn_flow_bninception.py | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/ucf101/tsn_2d_flow_bninception_seg3_f1s1_b32_g8-151870b7.pth) |


### Kinetics
| Model | Modality | Pretrained | Top-1 Acc. | Top-5 Acc. |                                     Config                                    |                                                                             Download                                                                         |
| :---: | :------: | :--------: | :--------: | :--------: | :---------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  I3D  |   RGB    |  ImageNet  |   72.9%    |   90.8%    | configs/kinetics400/i3d_kinetics400_3d_rgb_r50_c3d_inflate3x1x1_seg1_f32s2.py | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/i3d_kinetics_rgb_r50_c3d_inflated3x1x1_seg1_f32s2_f32s2-b93cc877.pth) |


### THUMOS14
| Model | Modality | Pretrained | mAP@0.10 | mAP@0.20 | mAP@0.30 | mAP@0.40 | mAP@0.50 |                      Config                       |                                                               Download                                                              |
| :---: | :------: | :--------: | :------: | :------: | :------: | :------: | :------: | :-----------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------: |
|  SSN  |    RGB   |  ImageNet  |  43.09%  |  37.95%  |  32.56%  |  25.71%  |  18.33%  | configs/thumos14/ssn_thumos14_rgb_bn_inception.py | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/thumos14/ssn_thumos14_rgb_bn_inception_tag-dac9ddb0.pth) |

### AVA
|   Model   | Pretrained | mAP@0.5 |                              Config                               |                                                                Download                                                              |
| :-------: | :--------: | :-----: | :---------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: |
| Fast-RCNN |  Kinetics  |  21.2   | configs/ava/ava_fast_rcnn_nl_r50_c4_1x_kinetics_pretrain_crop.py  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/ava/fast_rcnn_ava2.1_nl_r50_c4_1x_f32s2_kin-e2495b48.pth) |
