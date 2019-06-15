import logging
from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseRecognizer(nn.Module):
    """Base class for recognizers"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseRecognizer, self).__init__()

    @property
    def with_tenon_list(self):
        return hasattr(self, 'tenon_list') and self.tenon_list is not None

    @property
    def with_cls(self):
        return hasattr(self, 'cls_head') and self.cls_head is not None

    @abstractmethod
    def forward_train(self, num_modalities, **kwargs):
        pass

    @abstractmethod
    def forward_test(self, num_modalities, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info("load model from: {}".format(pretrained))

    def forward(self, num_modalities, img_meta, return_loss=True, **kwargs):
        num_modalities = int(num_modalities[0])
        if return_loss:
            return self.forward_train(num_modalities, img_meta, **kwargs)
        else:
            return self.forward_test(num_modalities, img_meta, **kwargs)
