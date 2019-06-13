import logging
from abc import ABCMeta, abstractmethod

import torch.nn as nn

class BaseLocalizer(nn.Module):
    """Base class for localizers"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseLocalizer, self).__init__()

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
