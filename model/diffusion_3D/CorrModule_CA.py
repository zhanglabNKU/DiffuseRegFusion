import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal
import torch.nn.functional as F


class RegHead(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.reg_head = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.reg_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.reg_head.weight.shape))
        self.reg_head.bias = nn.Parameter(torch.zeros(self.reg_head.bias.shape))

    def forward(self, x):
        x_out = self.reg_head(x)
        return x_out


class ResizeTransformer_block(nn.Module):
    def __init__(self, resize_factor, mode='bilinear'):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode

    def forward(self, x):
        if self.factor < 1:
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x
        elif self.factor > 1:
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
        return x


class SpatialTransformer_block(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        shape = flow.shape[2:]
        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)

        new_locs = grid + flow

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 1)
        # new_locs = new_locs[..., [2, 1, 0]]  # 去掉三维坐标转换

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class DualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x_out = self.act2(x)
        return x_out


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        x_out = self.act(x)
        return x_out


class Encoder(nn.Module):
    def __init__(self, in_channels=1, channel_num=8):
        super().__init__()
        self.conv_1 = DualConvBlock(in_channels, channel_num)
        self.conv_2 = DualConvBlock(channel_num, channel_num * 2)
        self.conv_3 = DualConvBlock(channel_num * 2, channel_num * 4)
        self.conv_4 = DualConvBlock(channel_num * 4, channel_num * 8)
        self.conv_5 = DualConvBlock(channel_num * 8, channel_num * 16)
        self.downsample = nn.AvgPool2d(2, stride=2)

    def forward(self, x_in):
        x_1 = self.conv_1(x_in)
        x = self.downsample(x_1)
        x_2 = self.conv_2(x)
        x = self.downsample(x_2)
        x_3 = self.conv_3(x)
        x = self.downsample(x_3)
        x_4 = self.conv_4(x)
        x = self.downsample(x_4)
        x_5 = self.conv_5(x)
        return [x_1, x_2, x_3, x_4, x_5]


class Channel_attention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(Channel_attention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channel, channel // ratio, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel // ratio, channel, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.gap(x)
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return out * x


class Spacial_attention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(Spacial_attention, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel // ratio, kernel_size=1, stride=1, padding=0)
        self.norm = nn.InstanceNorm2d(channel // ratio)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel // ratio, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return out * x


def get_winsize(x_size, window_size):
    use_window_size = list(window_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
    return tuple(use_window_size)


def window_partition(x_in, window_size):
    b, h, w, c = x_in.shape
    x = x_in.view(b,
                  h // window_size[0],
                  window_size[0],
                  w // window_size[1],
                  window_size[1],
                  c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)
    return windows


def window_reverse(windows, window_size, dims):
    b, h, w = dims
    x = windows.view(b,
                     h // window_size[0],
                     w // window_size[1],
                     window_size[0],
                     window_size[1],
                     -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class LocalCorrModule(nn.Module):
    def __init__(self, embed_dim, num_heads=8, window_size=[2, 2]):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.normx = nn.LayerNorm(embed_dim)
        self.normy = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.channel_attention = Channel_attention(embed_dim)
        self.spacial_attention = Spacial_attention(embed_dim)

    def forward(self, x_in, y_in, seg_img):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x_in = x_in.to(device)
        y_in = y_in.to(device)
        seg_img = seg_img.to(device)
        self.normx = self.normx.to(device)
        self.normy = self.normy.to(device)
        self.q_proj = self.q_proj.to(device)
        self.k_proj = self.k_proj.to(device)
        self.v_proj = self.v_proj.to(device)
        self.channel_attention = self.channel_attention.to(device)
        self.spacial_attention = self.spacial_attention.to(device)

        b, h, w, c = x_in.shape
        x = self.normx(x_in)
        y = self.normy(y_in)

        window_size = get_winsize((h, w), self.window_size)

        pad_l = pad_t = 0
        pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
        pad_b = (window_size[0] - h % window_size[0]) % window_size[0]

        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        y = nnf.pad(y, (0, 0, pad_l, pad_r, pad_t, pad_b))

        _, hp, wp, _ = x.shape
        dims = [b, hp, wp]

        x_windows = window_partition(x, window_size)
        y_windows = window_partition(y, window_size)

        b_, n_, c_ = x_windows.shape

        q = self.q_proj(x_windows)
        k = self.k_proj(y_windows)
        v = self.v_proj(y_windows)

        q = q.reshape(b_, n_, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b_, n_, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b_, n_, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        attn = attn @ v
        attn_windows = attn.permute(0, 2, 1, 3).reshape(b_, n_, c_)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))

        corr = window_reverse(attn_windows, window_size, dims[:3])

        if pad_r > 0 or pad_b > 0:
            corr = corr[:, :h, :w, :].contiguous()

        corr = corr.view(-1, h, w, self.embed_dim).permute(0, 3, 1, 2).contiguous()
        corr = self.channel_attention(corr)
        corr = self.spacial_attention(corr)
        corr = corr.permute(0, 2, 3, 1)
        return corr


class GlobalCorrModule(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=[10, 12]):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.n = embed_dim
        self.window_size = window_size
        self.normx = nn.LayerNorm(embed_dim)
        self.normy = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.channel_attention = Channel_attention(self.n)
        self.spacial_attention = Spacial_attention(self.n)

    def forward(self, x_in, y_in, seg_img):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x_in = x_in.to(device)
        y_in = y_in.to(device)
        seg_img = seg_img.to(device)
        self.normx = self.normx.to(device)
        self.normy = self.normy.to(device)
        self.q_proj = self.q_proj.to(device)
        self.k_proj = self.k_proj.to(device)
        self.v_proj = self.v_proj.to(device)
        self.channel_attention = self.channel_attention.to(device)
        self.spacial_attention = self.spacial_attention.to(device)

        b, h, w, c = x_in.shape
        x = self.normx(x_in)
        y = self.normy(y_in)

        cat1 = torch.cat([x, y], dim=3).permute(0, 3, 1, 2)

        x = x.reshape(b, -1, c)
        y = y.reshape(b, -1, c)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(b, -1, c)
        corr = out.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        corr = self.channel_attention(corr)
        corr = self.spacial_attention(corr)

        corr = corr.permute(0, 2, 3, 1)
        return corr