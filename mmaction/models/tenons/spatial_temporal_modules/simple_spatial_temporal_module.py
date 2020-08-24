import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registry import SPATIAL_TEMPORAL_MODULES


@SPATIAL_TEMPORAL_MODULES.register_module
class SimpleSpatialTemporalModule(nn.Module):
    def __init__(self, spatial_type='avg', spatial_size=7, temporal_size=1):
        super(SimpleSpatialTemporalModule, self).__init__()

        assert spatial_type in ['identity', 'avg', 'max']
        self.spatial_type = spatial_type

        self.spatial_size = spatial_size
        if spatial_size != -1:
            self.spatial_size = (spatial_size, spatial_size)

        self.temporal_size = temporal_size

        assert not (self.spatial_size == -1) ^ (self.temporal_size == -1)

        if self.temporal_size == -1 and self.spatial_size == -1:
            self.pool_size = (1, 1, 1)
            if self.spatial_type == 'avg':
                self.pool_func = nn.AdaptiveAvgPool3d(self.pool_size)
            if self.spatial_type == 'max':
                self.pool_func = nn.AdaptiveMaxPool3d(self.pool_size)
        else:
            self.pool_size = (self.temporal_size, ) + self.spatial_size
            if self.spatial_type == 'avg':
                self.pool_func = nn.AvgPool3d(self.pool_size, stride=1, padding=0)
            if self.spatial_type == 'max':
                self.pool_func = nn.MaxPool3d(self.pool_size, stride=1, padding=0)


    def init_weights(self):
        pass

    def forward(self, input):
        if self.spatial_type == 'identity':
            return input
        else:
            return self.pool_func(input)
