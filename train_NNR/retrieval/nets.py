####################################################################################################
# TANSmodels: Task-Adaptive Neural Network Search with Meta-Contrastive Learning
# Wonyong Jeong, Hayeon Lee, Geon Park, Eunyoung Hyung, Jinheon Baek, Sung Ju Hwang
# github: https://github.com/wyjeong/TANS, email: wyjeong@kaist.ac.kr
####################################################################################################

import math
import torch
import numpy as np
# from config import *
# from misc.utils import *
import torch.nn.functional as F
import torch.nn as nn

class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.proj_q = nn.Linear(in_ch, in_ch)
        self.proj_k = nn.Linear(in_ch, in_ch)
        self.proj_v = nn.Linear(in_ch, in_ch)
        self.proj = nn.Linear(in_ch, in_ch)
    def forward(self, x):
        B, C = x.shape
        h=x
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)
        q = q.view(B,1, C)
        k = k.view(B, C,1)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B,1,1]
        w = F.softmax(w, dim=-1)
        v = v.view(B,1, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, 1, C]
        h = h.view(B, C)
        h = self.proj(h)
        return x + h

class PerformancePredictor(torch.nn.Module):
    def __init__(self, args):
        super(PerformancePredictor, self).__init__()
        self.args = args
        self.fc = torch.nn.Linear(self.args.n_dims, 1)

    def forward(self, q, m):
        m = m.squeeze(1)
        p= m-q
        p = torch.sigmoid(self.fc(p))
        return p

class QueryEncoder(torch.nn.Module):
    def __init__(self, args):
        super(QueryEncoder, self).__init__()
        self.args = args
        self.fc = torch.nn.Linear(1000, self.args.n_dims)

    def forward(self, D):
        q = []
        for d in D:
            _q = self.fc(d)
            _q = torch.mean(_q, 0)
            _q = self.l2norm(_q.unsqueeze(0))
            q.append(_q)
        q = torch.stack(q).squeeze()
        return q

    def l2norm(self, x):
        norm2 = torch.norm(x, 2, dim=1, keepdim=True)
        x = torch.div(x, norm2)
        return x

class Swish(nn.Module):
    def forward(self, x):
        return  torch.relu(x)

class ConditionalEmbedding(nn.Module):
    def __init__(self, num_labels, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Linear(num_labels, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
    def forward(self, t):
        emb = self.condEmbedding(t.to(torch.float32))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, attn=True):
        super().__init__()
        self.block1 = nn.Sequential(
             Swish(),
            nn.Linear(in_ch, out_ch),
        )
        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(in_ch, out_ch),
        )
        self.act=Swish()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x, x1):
        h = self.block1(x)
        x1=self.cond_proj(x1)
        h=h-x1
        h = self.attn(h)
        return h


class ModelEncoder(torch.nn.Module):
    def __init__(self, args):
        super(ModelEncoder, self).__init__()
        self.args = args
        self.res1= ResBlock(1000,self.args.n_dims,attn=False)
    def l2norm(self, x):
        norm2 = torch.norm(x, 2, dim=1, keepdim=True)
        x = torch.div(x, norm2)
        return x
    def forward(self, v_f,rese):
        m = v_f
        m = F.normalize(m)
        m = self.res1(m,rese)
        m = self.l2norm(m)

        return m