# import torch
# import torch.nn as nn
# import numpy as np

# from typing import Optional, Sequence, Tuple, Union
# from monai.networks.layers.utils import get_act_layer, get_norm_layer
# from monai.networks.blocks.dynunet_block import get_conv_layer


# class UnetrBasicBlock(nn.Module):
#     """
#     A CNN module that can be used for UNETR, based on: "Hatamizadeh et al.,
#     UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
#     """

#     def __init__(
#             self,
#             spatial_dims: int,
#             in_channels: int,
#             out_channels: int,
#             kernel_size: Union[Sequence[int], int],
#             stride: Union[Sequence[int], int],
#             norm_name: Union[Tuple, str],
#             res_block: bool = False,
#             time_emb_dim: int = None,
#     ) -> None:  # -> None 表示这个函数不返回任何值
#         """
#         Args:
#             spatial_dims: number of spatial dimensions.
#             in_channels: number of input channels.
#             out_channels: number of output channels.
#             kernel_size: convolution kernel size.
#             stride: convolution stride.
#             norm_name: feature normalization type and arguments.
#             res_block: bool argument to determine if residual block is used.

#         """

#         super().__init__()  # 调用父类的__init__方法

#         if res_block:  # 一定会进入if而不会进入else段落
#             self.layer = UnetResBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 time_emb_dim=time_emb_dim,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 norm_name=norm_name,
#             )
#         else:
#             self.layer = UnetBasicBlock(  # type: ignore
#                 spatial_dims=spatial_dims,
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 norm_name=norm_name,
#             )

#     def forward(self, inp, time=None):
#         return self.layer(inp, time)


# class UnetrUpBlock(nn.Module):
#     """
#     An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
#     UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
#     """

#     def __init__(
#             self,
#             spatial_dims: int,
#             in_channels: int,
#             out_channels: int,
#             kernel_size: Union[Sequence[int], int],
#             upsample_kernel_size: Union[Sequence[int], int],
#             norm_name: Union[Tuple, str],
#             res_block: bool = False,
#             time_emb_dim: int = None,
#     ) -> None:
#         """
#         Args:
#             spatial_dims: number of spatial dimensions.
#             in_channels: number of input channels.
#             out_channels: number of output channels.
#             kernel_size: convolution kernel size.
#             upsample_kernel_size: convolution kernel size for transposed convolution layers.
#             norm_name: feature normalization type and arguments.
#             res_block: bool argument to determine if residual block is used.

#         """

#         super().__init__()
#         upsample_stride = upsample_kernel_size
#         self.transp_conv = get_conv_layer(
#             spatial_dims,
#             in_channels,
#             out_channels,
#             kernel_size=upsample_kernel_size,
#             stride=upsample_stride,
#             conv_only=True,
#             is_transposed=True,
#         )

#         if res_block:
#             self.conv_block = UnetResBlock(
#                 spatial_dims,
#                 out_channels + out_channels,
#                 out_channels,
#                 time_emb_dim=time_emb_dim,
#                 kernel_size=kernel_size,
#                 stride=1,
#                 norm_name=norm_name,
#             )
#         else:
#             self.conv_block = UnetBasicBlock(  # type: ignore
#                 spatial_dims,
#                 out_channels + out_channels,
#                 out_channels,
#                 kernel_size=kernel_size,
#                 stride=1,
#                 norm_name=norm_name,
#             )

#     def forward(self, inp, skip, time=None):
#         # number of channels for skip should equals to out_channels
#         out = self.transp_conv(inp)
#         out = torch.cat((out, skip), dim=1)
#         out = self.conv_block(out, time)
#         return out


# class UnetResBlock(nn.Module):
#     """
#     A skip-connection based module that can be used for DynUNet, based on:
#     `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
#     `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

#     Args:
#         spatial_dims: number of spatial dimensions.
#         in_channels: number of input channels.
#         out_channels: number of output channels.
#         kernel_size: convolution kernel size.
#         stride: convolution stride.
#         norm_name: feature normalization type and arguments.
#         act_name: activation layer type and arguments.
#         dropout: dropout probability.

#     """

#     def __init__(
#             self,
#             spatial_dims: int,
#             in_channels: int,
#             out_channels: int,
#             kernel_size: Union[Sequence[int], int],
#             stride: Union[Sequence[int], int],
#             norm_name: Union[Tuple, str],
#             act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
#             dropout: Optional[Union[Tuple, str, float]] = None,
#             time_emb_dim: int = None,
#     ):
#         super().__init__()  # 调用父类的__init__方法
#         self.conv1 = get_conv_layer(
#             spatial_dims,
#             in_channels,
#             out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             dropout=dropout,
#             conv_only=True,
#         )
#         self.conv2 = get_conv_layer(
#             spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True
#         )
#         self.lrelu = get_act_layer(name=act_name)
#         self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
#         self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
#         self.downsample = in_channels != out_channels  # 用于判断是否需要进行下采样
#         stride_np = np.atleast_1d(stride)  # 将stride转换为numpy数组
#         if not np.all(stride_np == 1):
#             self.downsample = True
#         if self.downsample:  # 为了残差连接时保证特征图的分辨率相同设置下采样部分
#             self.conv3 = get_conv_layer(
#                 spatial_dims, in_channels, out_channels, kernel_size=1, stride=stride, dropout=dropout, conv_only=True
#             )
#             self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

#         self.time_emb_dim = time_emb_dim
#         if time_emb_dim is not None:
#             self.time_mlp = nn.Sequential(
#                 nn.SiLU(),
#                 nn.Linear(time_emb_dim, out_channels)
#             )

#     def forward(self, inp, time=None):
#         # print('monai', inp.shape, time.shape)
#         residual = inp
#         out = self.conv1(inp)
#         out = self.norm1(out)
#         out = self.lrelu(out)
#         if self.time_emb_dim is not None:
#             # print("time emb")
#             out = out + self.time_mlp(time)[:, :, None, None]  # 扩展维度，将形状从[batch_size, out_channels]变为[batch_size, out_channels, 1, 1, 1]
#         out = self.conv2(out)
#         out = self.norm2(out)
#         if hasattr(self, "conv3"):  # 判断是否有conv3属性
#             residual = self.conv3(residual)
#         if hasattr(self, "norm3"):
#             residual = self.norm3(residual)
#         out += residual
#         out = self.lrelu(out)
#         return out


# class UnetrUpBlock(nn.Module):
#     """
#     An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
#     UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
#     """

#     def __init__(
#             self,
#             spatial_dims: int,
#             in_channels: int,
#             out_channels: int,
#             kernel_size: Union[Sequence[int], int],
#             upsample_kernel_size: Union[Sequence[int], int],
#             norm_name: Union[Tuple, str],
#             res_block: bool = False,
#             time_emb_dim: int = None,
#     ) -> None:
#         """
#         Args:
#             spatial_dims: number of spatial dimensions.
#             in_channels: number of input channels.
#             out_channels: number of output channels.
#             kernel_size: convolution kernel size.
#             upsample_kernel_size: convolution kernel size for transposed convolution layers.
#             norm_name: feature normalization type and arguments.
#             res_block: bool argument to determine if residual block is used.

#         """

#         super().__init__()
#         upsample_stride = upsample_kernel_size
#         self.transp_conv = get_conv_layer(
#             spatial_dims,
#             in_channels,
#             out_channels,
#             kernel_size=upsample_kernel_size,
#             stride=upsample_stride,
#             conv_only=True,
#             is_transposed=True,
#         )

#         if res_block:
#             self.conv_block = UnetResBlock(
#                 spatial_dims,
#                 out_channels + out_channels,
#                 out_channels,
#                 time_emb_dim=time_emb_dim,
#                 kernel_size=kernel_size,
#                 stride=1,
#                 norm_name=norm_name,
#             )
#         else:
#             self.conv_block = UnetBasicBlock(  # type: ignore
#                 spatial_dims,
#                 out_channels + out_channels,
#                 out_channels,
#                 kernel_size=kernel_size,
#                 stride=1,
#                 norm_name=norm_name,
#             )

#     def forward(self, inp, skip, time=None):
#         # number of channels for skip should equals to out_channels
#         out = self.transp_conv(inp)
#         out = torch.cat((out, skip), dim=1)
#         out = self.conv_block(out, time)
#         return out


import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Sequence, Tuple, Union
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.blocks.dynunet_block import get_conv_layer
from model.diffusion_3D.unet import SpatialTransform  


class UnetrBasicBlock(nn.Module):  # Encoder
    """
    A CNN module that can be used for UNETR, based on: "Hatamizadeh et al.,
    UNETR: Transformers for 2D Medical Image Segmentation <https://arxiv.org/abs/2103.10504 >"
    """
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            stride: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            res_block: bool = False,
            time_emb_dim: int = None,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions (set to 2).
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.
        """
        super().__init__()
        if res_block:
            self.layer = UnetResBlock(
                spatial_dims=2,  # 设置为2D
                in_channels=in_channels,
                out_channels=out_channels,
                time_emb_dim=time_emb_dim,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )
        else:
            self.layer = UnetBasicBlock(  # type: ignore
                spatial_dims=2,  # 设置为2D
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )

    def forward(self, inp, time=None):
        return self.layer(inp, time)


class UnetrUpBlock(nn.Module):  # Decoder
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 2D Medical Image Segmentation <https://arxiv.org/abs/2103.10504 >"
    """
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            res_block: bool = False,
            time_emb_dim: int = None,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions (set to 2).
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.
        """
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims=2,  # 设置为2D
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        if res_block:
            self.conv_block1 = UnetResBlock_up(
                spatial_dims=2,  # 设置为2D
                in_channels=out_channels,
                out_channels=out_channels,
                time_emb_dim=time_emb_dim,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
            self.conv_block2 = UnetResBlock(
                spatial_dims=2,  # 设置为2D
                in_channels=out_channels + out_channels + out_channels,
                out_channels=out_channels,
                time_emb_dim=time_emb_dim,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
            self.conv_block3 = UnetResBlock(
                spatial_dims=2,  # 设置为2D
                in_channels=out_channels + out_channels,
                out_channels=out_channels,
                time_emb_dim=time_emb_dim,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims=2,  # 设置为2D
                in_channels=out_channels + out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip, time=None, M_out=None):
        # number of channels for skip should equals to out_channels
        if M_out is not None:
            out1 = self.transp_conv(inp)
            out2, flow = self.conv_block1(out1, time)
            _, _, h, w = M_out.shape
            stn_size = [h, w]
            stn = SpatialTransform(stn_size).cuda()
            M_wrap = stn(M_out, flow)
            out3 = torch.cat((out1, skip, M_wrap), dim=1)
            out = self.conv_block2(out3, time)
        else:
            out = self.transp_conv(inp)
            out = torch.cat((out, skip), dim=1)
            out = self.conv_block3(out, time)
        return out


class UnetResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128 >`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486 >`_.
    """
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            stride: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout: Optional[Union[Tuple, str, float]] = None,
            time_emb_dim: int = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims=2,  # 设置为2D
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims=2,  # 设置为2D
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            conv_only=True,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=2, channels=out_channels)  # 设置为2D
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=2, channels=out_channels)  # 设置为2D
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims=2,  # 设置为2D
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                dropout=dropout,
                conv_only=True,
            )
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=2, channels=out_channels)  # 设置为2D
        self.time_emb_dim = time_emb_dim
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )

    def forward(self, inp, time=None):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        if self.time_emb_dim is not None:
            out = out + self.time_mlp(time)[:, :, None, None]  # 调整维度以适应2D
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


class UnetResBlock_up(nn.Module):
    """
    扩展的UnetResBlock，在原有输出的基础上，输出一个额外的特征图作为2D图像的变形场。
    """
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            stride: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout: Optional[Union[Tuple, str, float]] = None,
            time_emb_dim: int = None,
            flow_channels: int = 2,  # 默认为2，对应2D空间的x,y两个方向
    ):
        super().__init__()
        # 主要特征处理部分，与UnetResBlock相同
        self.conv1 = get_conv_layer(
            spatial_dims=2,  # 设置为2D
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims=2,  # 设置为2D
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            conv_only=True,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=2, channels=out_channels)  # 设置为2D
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=2, channels=out_channels)  # 设置为2D
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims=2,  # 设置为2D
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                dropout=dropout,
                conv_only=True,
            )
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=2, channels=out_channels)  # 设置为2D
        self.time_emb_dim = time_emb_dim
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        # 新增：用于生成变形场的卷积层
        self.flow_conv = get_conv_layer(
            spatial_dims=2,  # 设置为2D
            in_channels=out_channels,
            out_channels=flow_channels,  # 输出通道数为flow_channels，通常为2
            kernel_size=3,  # 使用3x3卷积核
            stride=1,
            dropout=None,
            conv_only=True,
        )
        # 可选：添加Tanh激活函数来限制变形场的范围
        self.flow_act = nn.Tanh()

    def forward(self, inp, time=None):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        if self.time_emb_dim is not None:
            out = out + self.time_mlp(time)[:, :, None, None]  # 调整维度以适应2D
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        # 生成变形场
        flow = self.flow_conv(out)
        flow = self.flow_act(flow)  # 使用Tanh限制变形场的范围在[-1,1]之间
        # 返回特征图和变形场
        return out, flow
