import torch
import torch.nn as nn
from ...registry import SEGMENTAL_CONSENSUSES

@SEGMENTAL_CONSENSUSES.register_module
class SimpleConsensus(nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(SimpleConsensus, self).__init__()
        assert consensus_type in ['avg']
        self.consensus_type = consensus_type
        self.dim = dim

    def init_weights(self):
        pass

    def forward(self, input):
        if self.consensus_type == 'avg':
            output = input.mean(dim=self.dim, keepdim=True)
        else:
            return None
        return output
