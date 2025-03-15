import torch
import math
import torch.nn as nn
from common import *
from einops import rearrange
from typing import Tuple
from torch import Tensor
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class MSDformer(nn.Module):
    def __init__(self, n_subs, n_ovls, n_colors, scale, n_feats, n_SSTM, conv=default_conv):
        super(MSDformer, self).__init__()
        self.head = MSAMG(n_subs, n_ovls, n_colors, n_feats)
        self.body = nn.ModuleList()
        self.N = n_SSTM
        self.middle = nn.ModuleList()
        for i in range(self.N):
            self.body.append(SSTM(n_feats, 6, False))
            self.middle.append(conv(n_feats, n_feats, 3))
        self.skip_conv = conv(n_colors, n_feats, 3)
        self.upsample = Upsampler(conv, scale, n_feats)
        self.tail = conv(n_feats, n_colors, 3)

    def forward(self, x, lms):
        x = self.head(x)
        xi = self.body[0](x)
        for i in range(1, self.N):
            xi = self.body[i](xi)
            xi = self.middle[i](xi)

        y = x + xi
        y = self.upsample(y)
        y = y + self.skip_conv(lms)
        y = self.tail(y)
        return y


class MSAMG(nn.Module):
    def __init__(self, n_subs, n_ovls, n_colors, n_feats, conv=default_conv):
        super(MSAMG, self).__init__()
        self.G = math.ceil((n_colors - n_ovls) / (n_subs - n_ovls))
        self.n_feats = n_feats
        # calculate group indices
        self.start_idx = []
        self.end_idx = []

        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > n_colors:
                end_ind = n_colors
                sta_ind = n_colors - n_subs
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)

        self.IG = MHSA(n_subs, n_feats)
        self.spc = nn.ModuleList()
        self.middle = nn.ModuleList()
        for n in range(self.G):
            self.spc.append(ResAttentionBlock(conv, n_feats, 1, res_scale=0.1))
            # self.spc.append(PSCA(n_feats, n_feats*2))
            self.middle.append(conv(n_feats, n_subs, 1))
        self.tail = conv(n_colors, n_feats, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        y = torch.zeros(b, c, h, w).cuda()
        channel_counter = torch.zeros(c).cuda()
        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]

            xi = x[:, sta_ind:end_ind, :, :]
            xi = self.IG(xi)
            xi = self.spc[g](xi)
            xi = self.middle[g](xi)
            y[:, sta_ind:end_ind, :, :] += xi
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1
        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        y = self.tail(y)
        return y


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, xf, fm):
        max_pool_f, _ = torch.max(xf, dim=1, keepdim=True)
        loc_pool_f = torch.mean(xf, dim=1, keepdim=True)
        max_pool_m, _ = torch.max(fm, dim=1, keepdim=True)
        loc_pool_m = torch.mean(fm, dim=1, keepdim=True)
        xf_att = self.LeakyReLU(self.conv(torch.cat([max_pool_f, loc_pool_f], dim=1)))
        fm_att = self.LeakyReLU(self.conv(torch.cat([max_pool_m, loc_pool_m], dim=1)))
        out_xf = xf_att * xf
        out_fm = fm_att * fm
        out = out_xf + out_fm

        return out



class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP)
    """

    def __init__(self, channel, conv=default_conv, bias=True):
        super().__init__()
        self.w_1 = nn.Conv2d(channel, channel, bias=bias, kernel_size=1)
        self.w_2 = nn.Conv2d(channel, channel, bias=bias, kernel_size=1)


    def forward(self, x):
        x1 = F.tanh(self.w_1(x))
        y = self.w_2(x1)
        return y


class PSA(nn.Module):
    """ Progressive Spectral Channel Attention (PSCA)
    """

    def __init__(self, d_model, d_ff, conv=default_conv):
        super().__init__()
        self.w_1 = conv(d_model, d_ff, 1)
        self.w_2 = conv(d_ff, d_model, 1)
        self.w_3 = conv(d_model, d_model, 1)

        nn.init.zeros_(self.w_3.weight)

    def forward(self, x):
        x = self.w_3(x) * x + x
        x = self.w_1(x)
        x = F.gelu(x)
        x = self.w_2(x)
        return x



class MHSA(nn.Module):
    """ Multi-Head Recurrent Spectral Attention
    """

    def __init__(self, channels, ffn_expansion_factor=2.66, multi_head=True, ffn=True, bias=False):
        super().__init__()
        self.channels = channels
        self.multi_head = multi_head
        self.ffn = ffn

        if ffn:
            self.ffn1 = MLP(channels)
            self.ffn2 = MLP(channels)

    def _conv_step(self, inputs):
        if self.ffn:
            Z = self.ffn1(inputs).tanh()
            F = self.ffn2(inputs).sigmoid()
        else:
            Z, F = inputs.split(split_size=self.channels, dim=1)
            Z, F = Z.tanh(), F.sigmoid()
        return Z, F

    def _rnn_step(self, z, f, h):
        h_ = (1 - f) * z if h is None else f * h + (1 - f) * z
        return h_

    def forward(self, inputs, reverse=True):
        Z, F = self._conv_step(inputs)

        if self.multi_head:
            Z1, Z2 = Z.split(self.channels // 2, 1)
            Z2 = torch.flip(Z2, [2])
            Z = torch.cat([Z1, Z2], dim=1)

            F1, F2 = F.split(self.channels // 2, 1)
            F2 = torch.flip(F2, [2])
            F = torch.cat([F1, F2], dim=1)

        h = None
        h_time = []

        if not reverse:
            for _, (z, f) in enumerate(zip(Z.split(1, 2), F.split(1, 2))):
                h = self._rnn_step(z, f, h)
                h_time.append(h)
        else:
            for _, (z, f) in enumerate((zip(
                    reversed(Z.split(1, 2)), reversed(F.split(1, 2))
            ))):  # split along timestep
                h = self._rnn_step(z, f, h)
                h_time.insert(0, h)

        y = torch.cat(h_time, dim=2)

        if self.multi_head:
            y1, y2 = y.split(self.channels // 2, 1)
            y2 = torch.flip(y2, [2])
            y = torch.cat([y1, y2], dim=1)

        return y


class DPB(nn.Module):
    def __init__(self, n_subs, n_feats, conv=default_conv):
        super(DPB, self).__init__()
        self.dconv1 = conv(n_subs, n_feats, 3, dilation=1)
        self.dconv2 = conv(n_subs, n_feats, 3, dilation=3)
        self.dconv3 = conv(n_subs, n_feats, 3, dilation=5)
        self.dconv4 = conv(n_subs, n_feats, 3, dilation=7)
        self.act = nn.PReLU()

    def forward(self, x):
        x1 = self.act(self.dconv1(x))
        # x2 = self.act(self.dconv2(x))
        # x3 = self.act(self.dconv3(x))
        x4 = self.act(self.dconv4(x))
        # y = x1 + x2 + x3
        y = x1 + x4
        # y = x3 + x1
        # y = x3
        return y


class MSSFM(nn.Module):
    def __init__(self, n_subs, n_feats, conv=default_conv):
        super(MSSFM, self).__init__()
        self.dpb = DPB(n_subs, n_feats)
        # self.spa = ResBlock(conv, n_feats, 3, res_scale=0.1)
        self.spa = MHSA(n_feats, n_feats)
        self.spe = PSA(n_feats, n_feats * 2)

    def forward(self, x):
        y1 = self.dpb(x)
        y2 = self.spa(y1)
        mul = torch.sigmoid(y2)
        y3 = self.spe(y1)
        y = y1 * mul + y3
        return y


class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """

    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor

            attn_logit：这是一个变量，用来存储计算得到的注意力得分（或称为注意力对数），即自注意力机制中的分数，这些分数将用于后续的注意力权重计算。

            query_hat：这是一个张量，代表查询（query）向量，它通常是通过某种变换（如线性变换）得到的，用于与键（key）向量进行比较。

            self.scale：这是一个缩放因子，用于调整查询向量的尺度。在某些自注意力实现中，为了防止点积过大导致梯度消失，会引入一个缩放因子，通常是键向量维度的平方根。

            @：这是PyTorch中的矩阵乘法运算符，相当于NumPy中的dot函数或@运算符。

            key_hat.transpose(-2, -1)：key_hat是另一个张量，代表键（key）向量。transpose(-2, -1)是对该张量进行转置操作，具体是对最后两个维度进行交换。在PyTorch中，-2和-1分别表示倒数第二个维度和最后一个维度。转置操作是为了使键向量与查询向量的维度对齐，以便进行点积运算。

            (query_hat*self.scale) @ key_hat.transpose(-2, -1)：这部分代码首先将查询向量query_hat乘以缩放因子self.scale，然后与转置后的键向量key_hat进行矩阵乘法，计算得到注意力得分。

            总结来说，这句代码是在自注意力机制中计算查询向量和键向量之间的点积，得到的结果attn_logit将用于后续的softmax操作，以生成每个位置的注意力权重。这些权重将用于加权值（value）向量，从而得到最终的输出。
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
            query_hat, key_hat = self.emb(query), self.emb(key)  # per-window pooling -> (n, p^2, c)
            attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)  # (n, p^2, p^2)
            topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # (n, p^2, k), (n, p^2, k)
            r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k)

        return r_weight, topk_index


class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
                               # (n, p^2, p^2, w^2, c_kv) without mem cpy
                               dim=2,
                               index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
                               # (n, p^2, k, w^2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv  # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        # else: #'none'
        #     topk_kv = topk_kv # do nothing

        return topk_kv


class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv
        # q, k, v = self.qkv(x).split([self.qk_dim, self.qk_dim, self.dim], dim=-1)
        # return q, k, v


class SSA(nn.Module):
    """
    n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
    kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
    topk: topk for window filtering
    param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
    param_routing: extra linear for routing
    diff_routing: wether to set routing differentiable
    soft_routing: wether to multiply soft routing weights
    """

    def __init__(self, dim, num_heads=8, n_win=8, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 side_dwconv=3,
                 auto_pad=False):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5

        ################side_dwconv (i.e. LCE in ShuntedTransformer)###########
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

        ################ global routing setting #################
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        ################ Adaptive Interaction Module #################
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )
        # router
        assert not (self.param_routing and not self.diff_routing)  # cannot be with_param=True and diff_routing=False
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        if self.soft_routing:  # soft routing, always diffrentiable (if no detach)
            mul_weight = 'soft'
        elif self.diff_routing:  # hard differentiable routing
            mul_weight = 'hard'
        else:  # hard non-differentiable routing
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')

        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity':  # no kv downsampling
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':
            # assert self.kv_downsample_ratio is not None
            # assert self.kv_downsample_kenel is not None
            # TODO: fracpool
            # 1. kernel size should be input size dependent
            # 2. there is a random factor, need to avoid independent sampling for k and v
            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':
            # TODO: need to consider the case where k != v so that need two downsample modules
            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')

        # softmax for local attention
        self.attn_act = nn.Softmax(dim=-1)

        self.auto_pad = auto_pad

    def forward(self, x, ret_attn_mask=False):
        """
        x: NHWC tensor

        Return:
            NHWC tensor
        """
        # NOTE: use padding for semantic segmentation
        ###################################################
        x = x.permute(0, 2, 3, 1)

        if self.auto_pad:
            N, H_in, W_in, C = x.size()

            pad_l = pad_t = 0
            pad_r = (8 - W_in % 8) % 8
            pad_b = (8 - H_in % 8) % 8
            x = F.pad(x, (0, 0,  # dim=-1
                          pad_l, pad_r,  # dim=-2
                          pad_t, pad_b))  # dim=-3
            _, H, W, _ = x.size()  # padded size
        else:
            N, H, W, C = x.size()
            assert H % 8 == 0 and W % 8 == 0  #
        ###################################################

        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=8, i=8)

        #################qkv projection###################
        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)
        # NOTE: separte kv if there were memory leak issue caused by gather
        q, kv = self.qkv(x)

        # pixel-wise qkv
        # q_pix: (n, p^2, w^2, c_qk)
        # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=8, i=8)

        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean(
            [2, 3])  # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)

        ##################side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp
        # lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=8, i=8).contiguous())
        lepe = self.dwconv(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=8, i=8).contiguous())
        # lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=8, i=8)

        ############ gather q dependent k/v #################

        r_weight, r_idx = self.router(q_win, k_win)  # both are (n, p^2, topk) tensors

        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)  # (n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
        # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
        # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)

        ######### do attention as normal ####################
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c',
                          m=self.num_heads)  # to BMLC tensor (n*p^2, m, w^2, c_qk//m)

        # param-free multihead attention
        attn_weight = (
                                  q_pix * self.scale) @ k_pix_sel  # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel  # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=8, i=8,
                        h=H // 8, w=W // 8)
        attention_reshape = out.permute(0, 3, 1, 2)
        channel_map = self.channel_interaction(attention_reshape)
        # S-Map (before sigmoid)
        spatial_map = self.spatial_interaction(lepe).permute(0, 2, 3, 1).contiguous().view(N, H * W, 1)

        # S-I
        out = out.view(N, H * W, C)
        out = out * torch.sigmoid(spatial_map)
        # C-I
        lepe = lepe * torch.sigmoid(channel_map)
        lepe = lepe.permute(0, 2, 3, 1).contiguous().view(N, H * W, C)

        out = out + lepe

        # out = out + lepe
        # output linear
        out = out.view(N, H, W, C)
        out = self.wo(out).permute(0, 3, 1, 2)

        # NOTE: use padding for semantic segmentation
        # crop padded region
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return out


class SSTM(nn.Module):
    """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            shift_size (int): Shift size for SW-MSA.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            drop (float, optional): Dropout rate. Default: 0.0
            drop_path (float, optional): Stochastic depth rate. Default: 0.0
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, shift_size=0, drop_path=0.0,
                 mlp_ratio=4., drop=0., act_layer=nn.GELU, bias=False):
        super(SSTM, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.num_heads = num_heads

        self.global_attn = SSA(dim, num_heads, bias)

    def forward(self, x):
        B, C, H, W = x.shape  # B, C, H*W
        x = x.flatten(2).transpose(1, 2)  # B, H*W, C
        shortcut = x
        x = self.norm1(x)

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # B C HW
        x = self.global_attn(x)  # global spectral self-attention

        x = x.flatten(2).transpose(1, 2)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.transpose(1, 2).view(B, C, H, W)
        return x

class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)

    def forward(self, x):
        return x + self.conv(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., conv_pos=True,
                 downsample=False, kernel_size=5):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        self.conv = ResDWC(hidden_features, 3)

    def forward(self, x):
        b, hw, c = x.size()
        hh = int(math.sqrt(hw))
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.conv(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)
        return x