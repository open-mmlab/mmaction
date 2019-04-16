import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registry import SPATIAL_TEMPORAL_MODULES


@SPATIAL_TEMPORAL_MODULES.register_module
class SimpleSpatialModule(nn.Module):
    def __init__(self, spatial_type='avg', spatial_size=7):
        super(SimpleSpatialModule, self).__init__()

        assert spatial_type in ['avg']
        self.spatial_type = spatial_type

        self.spatial_size = spatial_size if not isinstance(spatial_size, int) else (spatial_size, spatial_size)

        if self.spatial_type == 'avg':
            self.op = nn.AvgPool2d(self.spatial_size, stride=1, padding=0)


    def init_weights(self):
        pass

    def forward(self, input):
        return self.op(input)