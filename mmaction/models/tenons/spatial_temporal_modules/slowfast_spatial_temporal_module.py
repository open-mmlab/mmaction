import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registry import SPATIAL_TEMPORAL_MODULES


@SPATIAL_TEMPORAL_MODULES.register_module
class SlowFastSpatialTemporalModule(nn.Module):
    def __init__(self, adaptive_pool=True, spatial_type='avg', spatial_size=1, temporal_size=1):
        super(SlowFastSpatialTemporalModule, self).__init__()

        self.adaptive_pool = adaptive_pool
        assert spatial_type in ['avg']
        self.spatial_type = spatial_type

        self.spatial_size = spatial_size if not isinstance(spatial_size, int) else (spatial_size, spatial_size)
        self.temporal_size = temporal_size
        self.pool_size = (self.temporal_size, ) + self.spatial_size

        if self.adaptive_pool:
            if self.spatial_type == 'avg':
                self.op = nn.AdaptiveAvgPool3d(self.pool_size)
        else:
            raise NotImplementedError


    def init_weights(self):
        pass

    def forward(self, input):
        x_slow, x_fast = input
        x_slow = self.op(x_slow)
        x_fast = self.op(x_fast)
        return torch.cat((x_slow, x_fast), dim=1)
