import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LEAKY_ALPHA = 0.1


def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=LEAKY_ALPHA, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
        super(TemporalConv, self).__init__()

        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=(kernel_size, 1),
                              padding=(pad, 0),
                              stride=(stride, 1),
                              dilation=(dilation, 1),
                              groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class PointWiseTCN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(PointWiseTCN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class ST_GC(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(ST_GC, self).__init__()

        A = torch.from_numpy(A.astype(np.float32))
        self.A = nn.Parameter(A)
        self.Nh = A.size(0)

        self.conv = nn.Conv2d(in_channels, out_channels * self.Nh, 1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, C, T, V = x.size()
        v = self.conv(x).view(N, self.Nh, -1, T, V)
        weights = self.A.to(v.dtype)

        x = torch.einsum('hvu,nhctu->nctv', weights, v)
        x = self.bn(x)
        return x


class CTR_GC(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_scale=1):
        super(CTR_GC, self).__init__()

        A = torch.from_numpy(A.astype(np.float32))
        self.Nh = A.size(0)
        self.A = nn.Parameter(A)
        self.num_scale = num_scale

        rel_channels = in_channels // 8 if in_channels != 3 else 8

        self.conv1 = nn.Conv2d(in_channels, rel_channels * self.Nh, 1, groups=num_scale)
        self.conv2 = nn.Conv2d(in_channels, rel_channels * self.Nh, 1, groups=num_scale)
        self.conv3 = nn.Conv2d(in_channels, out_channels * self.Nh, 1, groups=num_scale)
        self.conv4 = nn.Conv2d(rel_channels * self.Nh, out_channels * self.Nh, 1, groups=num_scale * self.Nh)

        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)

        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(LEAKY_ALPHA)

    def forward(self, x, A=None, alpha=1):
        N, C, T, V = x.size()
        q, k, v = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x).view(N, self.num_scale, self.Nh, -1, T, V)
        weights = self.conv4(self.tanh(q.unsqueeze(-1) - k.unsqueeze(-2))).view(N, self.num_scale, self.Nh, -1, V, V)
        weights = weights * self.alpha.to(weights.dtype) + self.A.view(1, 1, self.Nh, 1, V, V).to(weights.dtype)
        x = torch.einsum('ngacvu, ngactu->ngctv', weights, v).contiguous().view(N, -1, T, V)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 stride=1,
                 dilations=[1, 2],
                 residual=False,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) is list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size=5, stride=1, dilations=2,
                 num_frame=64, num_joint=25, residual=True):
        super(TCN_GCN_unit, self).__init__()

        num_scale = 4
        self.num_scale = num_scale if in_channels != 3 else 1

        if in_channels == 3:
            self.gcn = ST_GC(in_channels, out_channels, A)
        else:
            self.gcn = CTR_GC(in_channels,
                              out_channels,
                              A,
                              self.num_scale)
        self.tcn = MultiScale_TemporalConv(out_channels, out_channels, stride=stride)

        if in_channels != out_channels:
            self.residual1 = PointWiseTCN(in_channels, out_channels, groups=self.num_scale)
        else:
            self.residual1 = lambda x: x

        if not residual:
            self.residual2 = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual2 = lambda x: x
        else:
            self.residual2 = PointWiseTCN(in_channels, out_channels, stride=stride, groups=self.num_scale)

        self.relu = nn.LeakyReLU(LEAKY_ALPHA)
        init_param(self.modules())

    def forward(self, x):
        res = x
        x = self.gcn(x)
        x = self.relu(x + self.residual1(res))
        x = self.tcn(x)
        x = self.relu(x + self.residual2(res))
        return x


class Fusion_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size=5, residual=True):
        super(Fusion_Block, self).__init__()

        self.gcn = ST_GC(in_channels, out_channels, A)
        self.tcn = TemporalConv(out_channels, out_channels, kernel_size)

        self.residual1 = PointWiseTCN(in_channels, out_channels)

        if not residual:
            self.residual2 = lambda x: 0
        else:
            self.residual2 = PointWiseTCN(in_channels, out_channels)

        self.relu = nn.LeakyReLU(LEAKY_ALPHA)
        init_param(self.modules())

    def forward(self, x):
        res = x
        x = self.gcn(x)
        x = self.relu(x + self.residual1(res))
        x = self.tcn(x)
        x = self.relu(x + self.residual2(res))
        return x
