import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registry import SEGMENTAL_CONSENSUSES

class _SimpleConsensus(torch.autograd.Function):
    """Simplest segmental consensus module"""

    consensus_type = 'avg'
    dim = 1
    shape = None

    @staticmethod
    def forward(ctx, x):
        _SimpleConsensus.shape = x.size()
        output = x.mean(dim=_SimpleConsensus.dim, keepdim=True)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_in = grad_output.expand(_SimpleConsensus.shape) / float(_SimpleConsensus.shape[_SimpleConsensus.dim])
        return grad_in


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
        _SimpleConsensus.consensus_type = self.consensus_type
        assert _SimpleConsensus.consensus_type in ['avg']
        _SimpleConsensus.dim = self.dim
        return _SimpleConsensus.apply(input)