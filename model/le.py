import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import TCN_GCN_unit, Fusion_Block


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class LEGCN(nn.Sequential):
    def __init__(self, block_args, A):
        super(LEGCN, self).__init__()
        for i, [in_channels, out_channels, stride, residual, num_frame, num_joint] in enumerate(block_args):
            self.add_module(f'block-{i}_tcngcn', TCN_GCN_unit(in_channels, out_channels, A, stride=stride, num_frame=num_frame, num_joint=num_joint, residual=residual))


class ProjectionHead(nn.Module):
    def __init__(self, emb_size, head_size):
        super(ProjectionHead, self).__init__()
        self.hidden = nn.Linear(emb_size, emb_size)
        self.out = nn.Linear(emb_size, head_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.hidden(h)
        h = F.relu_(h)
        h = self.out(h)
        return h


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, num_stream=2,
                 graph=None, graph_args=dict(), in_channels=3, label_dropout=0, instance_dropout=0, le_features=384, embedding_dir="", cal_feature_fc=False, cal_feature_label=False, cal_feature_label_att=False, cal_label_feature_fc=False, cal_label_feature_label=False, label_feature_pool='post', fused_x_proj=False, fused_emb_proj=False, base_frame=64):
        super(Model, self).__init__()

        assert cal_feature_label or cal_feature_label_att or cal_label_feature_fc or cal_label_feature_label

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64

        self.blockargs = [
            [in_channels, base_channel, 1, False, base_frame, num_point],
            [base_channel, base_channel, 1, True, base_frame, num_point],
            [base_channel, base_channel, 1, True, base_frame, num_point],
            [base_channel, base_channel, 1, True, base_frame, num_point],
            [base_channel, base_channel * 2, 2, True, base_frame, num_point],
            [base_channel * 2, base_channel * 2, 1, True, base_frame // 2, num_point],
            [base_channel * 2, base_channel * 2, 1, True, base_frame // 2, num_point],
            [base_channel * 2, base_channel * 4, 2, True, base_frame // 2, num_point],
            [base_channel * 4, base_channel * 4, 1, True, base_frame // 4, num_point],
            [base_channel * 4, base_channel * 4, 1, True, base_frame // 4, num_point]
        ]

        self.num_stream = num_stream
        self.streams = nn.ModuleList([LEGCN(self.blockargs, A) for _ in range(self.num_stream)])

        bn_init(self.data_bn, 1)

        if label_dropout:
            self.label_dropout = nn.Dropout(label_dropout)
        else:
            self.label_dropout = lambda x: x
        if instance_dropout:
            self.instance_dropout = nn.Dropout(instance_dropout)
        else:
            self.instance_dropout = lambda x: x

        self.cal_feature_fc = cal_feature_fc
        self.cal_feature_label = cal_feature_label
        self.cal_feature_label_att = cal_feature_label_att
        self.cal_label_feature_fc = cal_label_feature_fc
        self.cal_label_feature_label = cal_label_feature_label

        if self.cal_feature_fc:
            self.feature_fc = nn.ModuleList([nn.Linear(base_channel * 4, num_class) for _ in range(self.num_stream)])
            for fc in self.feature_fc:
                nn.init.normal_(fc.weight, 0, math.sqrt(2. / num_class))
        else:
            self.feature_fc = ['placeholder' for _ in range(self.num_stream)]

        if self.cal_label_feature_fc or self.cal_label_feature_label:
            self.fusion_blocks = nn.ModuleList([Fusion_Block(2, 1, A, residual=False) for _ in range(self.num_stream)])
            # self.fusion_blocks = nn.ModuleList([Fusion_Block(2, 1, A, residual=True) for _ in range(self.num_stream)])
            assert label_feature_pool == 'pre' or label_feature_pool == 'pre_fc' or label_feature_pool == 'post'
            self.label_feature_pool = label_feature_pool
            if self.label_feature_pool == 'pre_fc':
                # self.fusion_fcs = nn.ModuleList([ProjectionHead(self.num_class, self.num_class) for _ in range(self.num_stream)])
                self.fusion_fcs = nn.ModuleList([nn.Linear(self.num_class, self.num_class) for _ in range(self.num_stream)])
            else:
                self.fusion_fcs = ['placeholder' for _ in range(self.num_stream)]
            if self.cal_label_feature_fc:
                self.label_feature_fc = nn.ModuleList([nn.Linear(base_channel * 4, num_class) for _ in range(self.num_stream)])
                for fc in self.label_feature_fc:
                    nn.init.normal_(fc.weight, 0, math.sqrt(2. / num_class))
            else:
                self.label_feature_fc = ['placeholder' for _ in range(self.num_stream)]
        else:
            self.fusion_blocks = ['placeholder' for _ in range(self.num_stream)]
            self.fusion_fcs = ['placeholder' for _ in range(self.num_stream)]
            self.label_feature_fc = ['placeholder' for _ in range(self.num_stream)]

        self.le_features = le_features
        if not embedding_dir:
            self.label_embedding = nn.Embedding(self.num_class, self.le_features)
            nn.init.normal_(self.label_embedding.weight, 0, math.sqrt(2. / num_class))
            # self.label_embedding.weight.requires_grad = False
        else:
            loaded_arr = np.load(embedding_dir)
            self.le_features = loaded_arr.shape[1]
            loaded_arr = loaded_arr[:self.num_class]
            original_mean = np.mean(loaded_arr, axis=0)
            original_std = np.std(loaded_arr, axis=0)
            epsilon = 1e-8
            target_mean = 0
            target_std = math.sqrt(2. / num_class)
            loaded_arr = (loaded_arr - original_mean) / (original_std + epsilon) * target_std + target_mean
            embedding_tensor = torch.from_numpy(loaded_arr)
            self.label_embedding = nn.Embedding(self.num_class, self.le_features)
            with torch.no_grad():
                self.label_embedding.weight.copy_(embedding_tensor, non_blocking=True)
            self.label_embedding.weight.requires_grad = False
        self.proj_head = nn.ModuleList([ProjectionHead(self.le_features, base_channel * 4) for _ in range(self.num_stream)])
        if fused_x_proj and (self.cal_label_feature_fc or self.cal_label_feature_label):
            self.fused_x_proj = nn.ModuleList([ProjectionHead(base_channel * 4, base_channel * 4) for _ in range(self.num_stream)])
        else:
            self.fused_x_proj = [lambda x: x for _ in range(self.num_stream)]
        if fused_emb_proj and self.cal_feature_label_att:
            self.fused_emb_proj = nn.ModuleList([ProjectionHead(base_channel * 4, base_channel * 4) for _ in range(self.num_stream)])
        else:
            self.fused_emb_proj = [lambda x: x for _ in range(self.num_stream)]

    def cal_sim(self, z1, z2, num_heads, t, normalize):
        assert z1.shape[1] == z2.shape[1]
        assert z1.shape[1] % num_heads == 0
        head_dim = z1.shape[1] // num_heads
        z1 = z1.view(-1, num_heads, head_dim).transpose(0, 1)
        z2 = z2.view(-1, num_heads, head_dim).transpose(0, 1)
        if normalize:
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)
        sim_mat = torch.matmul(z1, z2.transpose(-2, -1)) / t

        return sim_mat

    def cal_att(self, x, emb, dim, num_heads=8, t=1.0, normalize=False):
        N, C, T, V = x.shape
        assert C == emb.shape[1]
        assert C % num_heads == 0
        x = x.mean(dim=dim)
        head_dim = C // num_heads
        x = x.view(N, num_heads, head_dim, -1).transpose(-2, -1)
        emb = emb.view(-1, num_heads, head_dim).transpose(0, 1)
        if normalize:
            x = F.normalize(x, dim=-1)
            emb = F.normalize(emb, dim=-1)
        sim_mat = torch.matmul(x, emb.transpose(-2, -1)) / t

        return sim_mat

    def fuse(self, x, pattern, fusion_block, fusion_fc, M, emb=None, t=1.0, normalize=False):
        # This module is not included in the current release due to subsequent research
        N, num_heads, T, V, num_class = pattern.shape

    def cal_feature_fc_score(self, x, fc, N, M):
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.instance_dropout(x)
        return fc(x)

    def cal_feature_label_score(self, x, emb, N, M, num_heads=8, t=1.0, normalize=False, mean_type='pre'):
        assert mean_type == 'pre' or mean_type == 'post'
        c_new = x.size(1)
        emb = self.label_dropout(emb)
        if mean_type == 'pre':
            x = x.view(N, M, c_new, -1)
            x = x.mean(3).mean(1)
            x = self.instance_dropout(x)
            return self.cal_sim(x, emb, num_heads, t, normalize).transpose(0, 1)
        elif mean_type == 'post':
            x = x.view(N * M, c_new, -1)
            x = x.mean(2)
            x = self.instance_dropout(x)
            x = self.cal_sim(x, emb, num_heads, t, normalize).transpose(0, 1)
            return x.reshape(N, M, num_heads, -1).mean(1)

    def feature_label_att_fuse(self, x, emb, N, M, num_heads=8, t=1.0, normalize=False):
        c_new = x.size(1)
        x = x.view(N * M, c_new, -1)
        x = x.mean(2)
        scores = self.cal_sim(x, emb, num_heads, t, normalize).transpose(0, 1)
        scores = F.softmax(scores, dim=-1)
        num_class = emb.shape[0]
        emb = emb.reshape(num_class, num_heads, -1).permute(1, 2, 0)
        fused_emb = torch.einsum('nkc,kdc->nkd', scores, emb).reshape(N * M, -1)
        return fused_emb.reshape(N, M, -1)

    def cal_label_feature_fc_score(self, x, fc):
        x = x.mean(1)
        x = self.instance_dropout(x)
        return fc(x)

    def cal_label_feature_label_score(self, x, emb, num_heads=8, t=1.0, normalize=False, mean_type='pre'):
        assert mean_type == 'pre' or mean_type == 'post'
        emb = self.label_dropout(emb)
        if mean_type == 'pre':
            x = x.mean(1)
            x = self.instance_dropout(x)
            return self.cal_sim(x, emb, num_heads, t, normalize).transpose(0, 1)
        elif mean_type == 'post':
            N, M, _ = x.shape
            x = x.view(N * M, -1)
            x = self.instance_dropout(x)
            x = self.cal_sim(x, emb, num_heads, t, normalize).transpose(0, 1)
            return x.reshape(N, M, num_heads, -1).mean(1)

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        out_feature_fc_score = []
        out_label_feature_fc_score = []
        out_label_feature_label_score = []
        out_feature_label_score = []
        out_fused_x = []
        out_fused_emb = []

        x_ = x
        index = torch.arange(self.num_class).long().to(x.device)
        label_emb = self.label_embedding(index)
        for stream, proj_head, fusion_block, fusion_fc, feature_fc, label_feature_fc, fused_x_proj, fused_emb_proj in zip(self.streams, self.proj_head, self.fusion_blocks, self.fusion_fcs, self.feature_fc, self.label_feature_fc, self.fused_x_proj, self.fused_emb_proj):
            x = x_
            x = stream(x)
            label_emb_ = proj_head(label_emb)

            if self.cal_feature_fc:
                feature_fc_score = self.cal_feature_fc_score(x, feature_fc, N, M)
                out_feature_fc_score.append(feature_fc_score)
            else:
                out_feature_fc_score.append('placeholder')

            if self.cal_label_feature_fc or self.cal_label_feature_label:
                att_v = self.cal_att(x, label_emb_, dim=2, num_heads=8)
                att_t = self.cal_att(x, label_emb_, dim=3, num_heads=8)
                pattern = att_v.unsqueeze(2) + att_t.unsqueeze(3)  # N, num_heads, T, V, num_class
                # fused_x = self.fuse(x, pattern, fusion_block, fusion_fc, M, label_emb_)
                fused_x = self.fuse(x, pattern, fusion_block, fusion_fc, M)
                out_fused_x.append(self.instance_dropout(fused_x_proj(fused_x)))
                if self.cal_label_feature_fc:
                    label_feature_fc_score = self.cal_label_feature_fc_score(fused_x, label_feature_fc)
                    out_label_feature_fc_score.append(label_feature_fc_score)
                else:
                    out_label_feature_fc_score.append('placeholder')
                if self.cal_label_feature_label:
                    label_feature_label_score = self.cal_label_feature_label_score(fused_x, label_emb_, num_heads=1)
                    out_label_feature_label_score.append(label_feature_label_score)
                else:
                    out_label_feature_label_score.append('placeholder')
            else:
                out_fused_x.append('placeholder')
                out_label_feature_fc_score.append('placeholder')
                out_label_feature_label_score.append('placeholder')

            if self.cal_feature_label:
                feature_label_score = self.cal_feature_label_score(x, label_emb_, N, M, num_heads=1)
                out_feature_label_score.append(feature_label_score)
            else:
                out_feature_label_score.append('placeholder')

            if self.cal_feature_label_att:
                fused_emb = self.feature_label_att_fuse(x, label_emb_, N, M, num_heads=8)
                out_fused_emb.append(self.label_dropout(fused_emb_proj(fused_emb)))
            else:
                out_fused_emb.append('placeholder')

        return out_feature_fc_score, out_label_feature_fc_score, out_label_feature_label_score, out_feature_label_score, out_fused_x, out_fused_emb
