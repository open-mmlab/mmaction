from .two_stage import TwoStageDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class FastRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 dropout_ratio=0,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(FastRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            bbox_roi_extractor=bbox_roi_extractor,
            dropout_ratio=dropout_ratio,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
