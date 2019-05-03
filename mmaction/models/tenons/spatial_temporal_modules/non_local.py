import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from ...registry import SPATIAL_TEMPORAL_MODULES


@SPATIAL_TEMPORAL_MODULES.register_module
class NonLocalModule(nn.Module):
    def __init__(self, in_channels=1024, nonlocal_type="gaussian", dim=3, embed=True, embed_dim=None, sub_sample=True, use_bn=True):
        super(NonLocalModule, self).__init__()

        assert nonlocal_type in ['gaussian', 'dot', 'concat']
        assert dim == 2 or dim == 3
        self.nonlocal_type = nonlocal_type
        self.embed = embed
        self.embed_dim = embed_dim if embed_dim is not None else in_channels // 2
        self.sub_sample = sub_sample
        self.use_bn = use_bn

        if self.embed:
            if dim == 2:
                self.theta = nn.Conv2d(in_channels, self.embed_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
                self.phi = nn.Conv2d(in_channels, self.embed_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
                self.g = nn.Conv2d(in_channels, self.embed_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
            elif dim == 3:
                self.theta = nn.Conv3d(in_channels, self.embed_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
                self.phi = nn.Conv3d(in_channels, self.embed_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
                self.g = nn.Conv3d(in_channels, self.embed_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

        if self.nonlocal_type == 'gaussian':
            self.softmax = nn.Softmax(dim=2)
        elif self.nonlocal_type == 'concat':
            if dim == 2:
                self.concat_proj = nn.Sequential(nn.Conv2d(self.embed_dim * 2, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                                                 nn.ReLU())
            elif dim == 3:
                self.concat_proj = nn.Sequential(nn.Conv3d(self.embed_dim * 2, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
                                                 nn.ReLU())

        if sub_sample:
            if dim == 2:
                self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
            elif dim == 3:
                self.max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
            self.g = nn.Sequential(self.max_pool, self.g)
            self.phi = nn.Sequential(self.max_pool, self.phi)

        if dim == 2:
            self.W = nn.Conv2d(self.embed_dim, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)) 
        elif dim == 3:
            self.W = nn.Conv3d(self.embed_dim, in_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)) 

        if use_bn:
            if dim == 2:
                self.bn = nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.9, affine=True)
            elif dim == 3:
                self.bn = nn.BatchNorm3d(in_channels, eps=1e-05, momentum=0.9, affine=True)
            self.W = nn.Sequential(self.W, self.bn)

 
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
               kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
               constant_init(m, 0)


    def forward(self, input):
        if self.embed:
            theta = self.theta(input)
            phi = self.phi(input)
            g = self.g(input)
        else:
            theta = input
            phi = input
            g = input
        
        if self.nonlocal_type in ['gaussian', 'dot']:
            # reshape [BxC'xTxHxW] to [BxC'x(T)HW]
            theta = theta.reshape(theta.shape[:2] + (-1,))
            phi = phi.reshape(theta.shape[:2] + (-1,))
            g = g.reshape(theta.shape[:2] + (-1,))
            theta_phi = torch.matmul(theta.transpose(1, 2), phi)
            if self.nonlocal_type == 'gaussian':
                p = self.softmax(theta_phi)
            elif self.nonlocal_type == 'dot':
                N = theta_phi.size(-1)
                p = theta_phi / N
        elif self.non_local_type == 'concat':
            # reshape [BxC'xTxHxW] to [BxC'x(T)HWx1]
            theta = theta.reshape(theta.shape[:2] + (-1,1))
            # reshape [BxC'xTxHxW] to [BxC'x1x(T)HW]
            phi = phi.reshape(theta.shape[:2] + (1,-1))
            theta_x = theta.repeat(1, 1, 1, phi.size(3))
            phi_x = phi.repeat(1, 1, theta.size(2), 1)
            theta_phi = torch.cat([theta_x, phi_x], dim=1)
            theta_phi = self.concat_proj(theta_phi)
            theta_phi = theta_phi.squeeze()
            N = theta_phi.size(-1)
            p = theta_phi / N
        else:
            NotImplementedError

        # BxC'xddd , Bxdxddd => BxC'xd
        y = torch.matmul(g, p.transpose(1, 2))
        y = y.reshape(y.shape[:2] + input.shape[2:])
        z = self.W(y) + input 

        return z
        

