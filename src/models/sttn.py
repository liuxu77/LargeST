import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base.model import BaseModel

class STTN(BaseModel):
    '''
    Reference code: https://github.com/xumingxingsjtu/STTN
    '''
    def __init__(self, device, supports, blocks, mlp_expand, hidden_channels, end_channels, dropout, **args):
        super(STTN, self).__init__(**args)
        self.t_modules = nn.ModuleList()
        self.s_modules = nn.ModuleList()
        self.bn = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=self.input_dim,
                                    out_channels=hidden_channels,
                                    kernel_size=(1, 1))

        self.supports = supports
        self.blocks = blocks
        for b in range(blocks):
            self.t_modules.append(
                TemporalTransformer(dim=hidden_channels,
                                    depth=1, heads=4,
                                    mlp_dim=hidden_channels * mlp_expand,
                                    time_num=self.seq_len,
                                    dropout=dropout,
                                    window_size=self.seq_len,
                                    device=device,
                                    ))

            self.s_modules.append(
                SpatialTransformer(dim=hidden_channels,
                                   depth=1, heads=4,
                                   mlp_dim=hidden_channels * mlp_expand,
                                   node_num=self.node_num,
                                   dropout=dropout,
                                   stage=b,
                                   ))

            self.bn.append(nn.BatchNorm2d(hidden_channels))

        self.end_conv_1 = nn.Conv2d(in_channels=hidden_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=self.output_dim * self.horizon,
                                    kernel_size=(1, 1),
                                    bias=True)


    def forward(self, inputs, label=None):  # (b, t, n, f)
        x = inputs.transpose(1, 3)
        x = self.start_conv(x)
        for i in range(self.blocks):
            residual = x
            x = self.s_modules[i](x, torch.stack(self.supports))
            x = self.t_modules[i](x)
            x = self.bn[i](x) + residual

        x = x[..., -1:]
        out = F.relu(self.end_conv_1(x))
        out = self.end_conv_2(out)
        return out


class TemporalTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, time_num, dropout, window_size, device):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, time_num, dim))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                TemporalAttention(dim=dim, heads=heads,
                                  window_size=window_size,
                                  dropout=dropout,
                                  causal=True,
                                  stage=i,
                                  device=device),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))


    def forward(self, x):
        b, c, n, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b*n, t, c)
        x = x + self.pos_embedding
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.reshape(b, n, t, c).permute(0, 3, 1, 2)
        return x


class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=8, window_size=1, dropout=0., causal=True, stage=0, device=None, qkv_bias=False, qk_scale=None):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.num_heads = heads
        self.causal = causal
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size
        self.stage = stage

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        self.mask = torch.tril(torch.ones(window_size, window_size)).to(device)


    def forward(self, x):
        B_prev, T_prev, C_prev = x.shape
        if self.window_size > 0:
            x = x.reshape(-1, self.window_size, C_prev)
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.causal:
            attn = attn.masked_fill_(
                self.mask == 0, float("-inf")).softmax(dim=-1)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if self.window_size > 0:
            x = x.reshape(B_prev, T_prev, C_prev)
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, node_num, dropout, stage=0):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, node_num, dim))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                SpatialAttention(dim, heads=heads,
                                 dropout=dropout,
                                 stage=stage),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                GCN(dim, dim, dropout, support_len=2),
            ]))


    def forward(self, x, adj):
        b, c, n, t = x.shape
        x = x.permute(0, 3, 2, 1).reshape(b*t, n, c)
        x = x + self.pos_embedding
        for attn, ff, gcn in self.layers:
            residual = x.reshape(b, t, n, c)
            x = attn(x, adj) + x
            x = ff(x) + x

            x = gcn(residual.permute(0, 3, 2, 1), adj).permute(
                0, 3, 2, 1).reshape(b*t, n, c) + x
        x = x.reshape(b, t, n, c).permute(0, 3, 2, 1)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0., stage=0, qkv_bias=False, qk_scale=None):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.stage = stage 

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)


    def forward(self, x, adj=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn


    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        return self.net(x)


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()


    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)


    def forward(self,x):
        return self.mlp(x)

    
class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order


    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h