import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck

from traj_conv import TrajConv
 
num_deformable_groups = 2

N, inC, inT, inH, inW = 2, 8, 8, 4, 4
outC, outT, outH, outW = 4, 8, 4, 4
kT, kH, kW = 3, 3, 3

conv = nn.Conv3d(inC, num_deformable_groups * 3 * kT * kH * kW,
                 kernel_size=(kT, kH, kW),
                 stride=(1,1,1),
                 padding=(1,1,1),
                 bias=False)

conv_offset3d = TrajConv(inC, outC, (kT, kH, kW),
                         stride=(1,1,1), padding=(1,1,1),
                         num_deformable_groups=num_deformable_groups).double().cuda()
 
input = torch.randn(N, inC, inT, inH, inW, requires_grad=True).double().cuda()
offset = torch.rand(N, num_deformable_groups * 2 * kT * kH * kW, inT, inH, inW, requires_grad=True) * 1 - 0.5
offset = offset.double().cuda()
test = gradcheck(conv_offset3d, (input, offset), eps=1e-5, atol=1e-1, rtol=1e-5)
print(test)
