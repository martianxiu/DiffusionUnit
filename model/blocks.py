import numpy as np
from lib.pointops.functions import pointops
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy



def block_decider(name):
    if name == 'dw_kpconv':
        return DepthwiseKPConv
    if name == 'residual':
        return ResidualBlock
    if name == 'upsample':
        return Upsampling 
    if name == 'downsample':
        return Downsampling 
    if name == 'simple':
        return SimpleBlock 
    if name == 'diffusion_unit':
        return DiffusionUnit 

# blocks 
class SimpleBlock(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride, config):
        super().__init__()
        func = config.convolution
        self.func = block_decider(func)(d_in, d_out, config)
        self.nsample = nsample
    
    def forward(self, p, x, o):
        N, C = x.size()
        pj_xj = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True)  # (m, nsample, 3+c)
        pj, xj = pj_xj[:, :, 0:3], pj_xj[:, :, 3:]
        x = self.func(p, pj, x, xj)
        return p, x, o
 
   
class ResidualBlock(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride, config):
        super().__init__()
        func = config.convolution
        bottleneck_ratio = config.bottleneck_ratio
        if bottleneck_ratio is None:
            self.reduction = self.expansion = nn.Identity()
            self.func = block_decider(func)(d_in, d_out, config)
        else:
            d_mid = d_in // bottleneck_ratio
            self.reduction = nn.Sequential(
                nn.Linear(d_in, d_mid),
                nn.BatchNorm1d(d_mid),
                nn.ReLU(inplace=True)
            )
            self.func = block_decider(func)(d_mid, d_mid, config)
            self.expansion = nn.Sequential(
                nn.Linear(d_mid, d_out),
                nn.BatchNorm1d(d_out),
                nn.ReLU(inplace=True)
            )
        self.nsample = nsample
    
    def forward(self, p, x, o):
        N, C = x.size()
        identity = x
        x = self.reduction(x)
        pj_xj = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True)  # (m, nsample, 3+c)
        pj, xj = pj_xj[:, :, 0:3], pj_xj[:, :, 3:]
        x = self.func(p, pj, x, xj)
        x = self.expansion(x)
        x = identity + x
        return p, x, o
        

class Downsampling(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride, config):
        super().__init__()
        self.d_in = d_in
        self.nsample = 16 
        self.stride = stride 
        self.mlp = nn.Sequential(
            nn.Linear(d_in+3, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, p, x, o):
        identity = x

        n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
        for i in range(1, o.shape[0]):
            count += (o[i].item() - o[i-1].item()) // self.stride
            n_o.append(count)
        n_o = torch.cuda.IntTensor(n_o)
        idx = pointops.furthestsampling(p, o, n_o)  # (m)
        n_p = p[idx.long(), :]  # (m, 3)
        
        pj_xj = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  
        pj, xj = pj_xj[:, :, :3], pj_xj[:, :, 3:]
        pj = pj / (torch.max(torch.norm(pj, dim=-1, keepdim=True), dim=1, keepdim=True)[0] + 1e-8)
        pj_xj = torch.cat([pj, xj], dim=-1)
        x = self.mlp(pj_xj.max(1)[0])
        return n_p, x, n_o 


class Upsampling(nn.Module):
    def __init__(self, d_in_sparse_dense, d_out, nsample, stride, config):
        super().__init__()
        d_in_sparse, d_in_dense = d_in_sparse_dense
        self.nsample = nsample
        self.d_out = d_out

        self.mlp = nn.Sequential(
            nn.Linear(d_in_sparse+ d_in_dense, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, p1,x1,o1, p2,x2,o2):
        '''
            pxo1: dense 
            pxo2: sparse  
        '''
        interpolated = pointops.interpolation(p2, p1, x2, o2, o1)
        x = self.mlp(torch.cat([x1, interpolated], dim=1))
        return p1, x, o1

# Convolutions
class DepthwiseKPConv(nn.Module):
    def __init__(self, d_in, d_out, config):
        super().__init__()
        if d_in == d_out:
            d_mid = d_in
            num_group = d_mid
            self.first_layer = False
        else:
            d_in = d_in + 3
            num_group = 1
            self.first_layer = True
        kernel_point = np.load('../model/kernels/dispositions/k_015_center_3D.npy').reshape(1,-1, 3) # (1, n_k, 3)

        num_kernel = kernel_point.shape[1]
        self.sigma = 0.3
        self.scale = (self.sigma ** 2) * 2 + 1e-10

        self.kernel_point = nn.Parameter(
            torch.tensor(kernel_point, dtype=torch.float32),
            requires_grad=False
        )
         
        self.depthwise = nn.Sequential(
            nn.Conv1d(d_in, d_out, num_kernel, groups=num_group),
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, p, pj, x, xj):
        l2_dist = torch.norm(pj, p=2, dim=2, keepdim=True) 
        pj = pj / (torch.max(l2_dist, dim=1, keepdim=True)[0] + 1e-10)
        sqr_dist = self.kernel_point[:, :, None, :] - pj[:, None, :, :] # (n, n_k, n_pj, c) 
        sqr_dist = (sqr_dist ** 2).sum(3) # (n, n_k, n_pj)
        corr = torch.exp(-sqr_dist/self.scale)
        if self.first_layer:
            xj = torch.cat([pj, xj], dim=-1)
        x = torch.matmul(corr, xj) # (n, n_k, c)
        x = self.depthwise(x.permute(0, 2, 1).contiguous())
        x = x.squeeze(2)
        return x

# Units
class DiffusionUnit(nn.Module):
    '''
        A slightly more efficient and mathematically equivalent implementation of Diffusion Unit
    '''
    def __init__(self, d_in, d_out, nsample, stride, config):
        super().__init__()
        self.nsample = nsample
        self.pre_linear = nn.Linear(d_in, d_in)
        self.activation = nn.ReLU(inplace=True)
        self.varphi = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.BatchNorm1d(d_in),
            nn.ReLU(inplace=True)
        )   
    def forward(self, p, u, o):
        N, C = u.shape
        u_t = u
        u = self.pre_linear(u)
        u_n = pointops.queryandgroup(self.nsample, p, p, u, None, o, o, use_xyz=False) # (n,nsample,c)

        nabla_u = u_n - u.unsqueeze(1) # (n, nsample, c)
        nabla_u = self.activation(nabla_u)
        u_tt = self.varphi(nabla_u.mean(1)) + u_t 

        return p, u_tt, o 
