import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.utils import OptConfigType


class Conv2dGNGELU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.GroupNorm(4, out_channel),
            nn.GELU()
        )


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, dilation=1, norm_type='gn', gn_num=4):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size, stride, padding=padding, dilation=dilation, groups=dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

        if norm_type == 'gn':
            self.norm_layer = nn.GroupNorm(gn_num, dim_out)
        elif norm_type == 'bn':
            self.norm_layer = nn.BatchNorm2d(dim_out)
        else:
            self.norm_layer = nn.BatchNorm2d(dim_out)

        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm_layer(x)
        return self.activation(x)


class InvertedDepthWiseConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, expend_ration=2):
        super().__init__()
        hidden_channel = in_channel * expend_ration
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        layers.append(Conv2dGNGELU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            Conv2dGNGELU(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=stride, groups=hidden_channel),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.GroupNorm(4, out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        self.a = nn.Parameter(torch.tensor(0.6))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature1, feature2):
        fused_feature = self.a * feature1 + (1 - self.a) * feature2
        fused_feature = self.relu(fused_feature)
        return fused_feature


class SoftPool_2D(nn.Module):
    def __init__(self, kernel_size, stride):
        super(SoftPool_2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def soft_pool2d_horizontal(self, x, kernel_size=(1, 1), stride=(1, 1)):
        _, c, h, w = x.size()
        e_x = torch.sum(torch.exp(x), dim=1, keepdim=True)
        return F.avg_pool2d(x.mul(e_x), kernel_size, stride=stride).mul_(kernel_size[1]).div_(
            F.avg_pool2d(e_x, kernel_size, stride=stride).mul_(kernel_size[1]))

    def soft_pool2d_vertical(self, x, kernel_size=(1, 1), stride=(1, 1)):
        _, c, h, w = x.size()
        e_x = torch.sum(torch.exp(x), dim=1, keepdim=True)
        return F.avg_pool2d(x.mul(e_x), kernel_size, stride=stride).mul_(kernel_size[0]).div_(
            F.avg_pool2d(e_x, kernel_size, stride=stride).mul_(kernel_size[0]))

    def forward(self, x):
        _, _, h, w = x.size()
        horizontal_x = self.soft_pool2d_horizontal(x, kernel_size=(1, w), stride=(1, w))
        vertical_z = self.soft_pool2d_vertical(x, kernel_size=(h, 1), stride=(h, 1))
        return horizontal_x, vertical_z

class SFAM(nn.Module):
    def __init__(self, dim, reduction_ratio=4):
        super().__init__()
        # Expanded receptive field with dilated convolution
        self.x_conv3 = DepthWiseConv2d(dim, dim // 2, kernel_size=3, stride=1, padding=2, dilation=2)

        # Multi-scale feature extraction with different dilation rates
        self.branch1 = DepthWiseConv2d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1, dilation=1)
        self.branch2 = DepthWiseConv2d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=3, dilation=3)

        # Channel attention for adaptive feature recalibration
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim // 2, dim // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction_ratio, dim // 2, kernel_size=1),
            nn.Sigmoid()
        )

        # Enhanced spatial pooling
        self.soft_gap = SoftPool_2D(2, 2)

        # Position-sensitive feature fusion
        self.x_conv1 = DepthWiseConv2d(dim // 2, dim, kernel_size=1, stride=1, padding=0, dilation=1)

        # Residual connection
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Feature extraction with dilated convolution
        out_dw = self.x_conv3(x)

        # Multi-scale feature extraction
        out_branch1 = self.branch1(out_dw)
        out_branch2 = self.branch2(out_dw)
        out_multi = torch.cat([out_branch1, out_branch2], dim=1)

        # Apply channel attention
        channel_attn = self.channel_attention(out_multi)
        out_multi = out_multi * channel_attn

        # Spatial context modeling with softpool
        out_u_h, out_u_w = self.soft_gap(out_multi)

        # Cross-dimensional interaction
        out_spatial = out_u_w * out_u_h

        # Position-sensitive feature fusion
        out = self.x_conv1(out_spatial)

        # Upsample to original size
        out = F.interpolate(out, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=True)

        # Residual connection for gradient flow
        out = self.gamma * out + x

        return out

class FFSL(nn.Module):
    def __init__(self, dim, bias=False, a=16, b=16, c_h=16, c_w=16):
        super().__init__()
        self.x_conv3 = nn.Conv2d(dim, dim // 2, kernel_size=3, padding=2, dilation=2)
        self.soft_gap = SoftPool_2D(2, 2)
        self.x_conv1 = nn.Conv2d(dim // 2, dim, kernel_size=1)

        # 将 dim a b c_h c_w 五个参数注册为不可训练的缓冲区
        self.register_buffer("dim_val", torch.tensor(dim, dtype=torch.int32))
        self.register_buffer("a_val", torch.tensor(a, dtype=torch.int32))
        self.register_buffer("b_val", torch.tensor(b, dtype=torch.int32))
        self.register_buffer("c_h_val", torch.tensor(c_h, dtype=torch.int32))
        self.register_buffer("c_w_val", torch.tensor(c_w, dtype=torch.int32))

        # 创建四个可训练的权重参数
        self.a_weight = nn.Parameter(torch.ones(2, 1, dim // 4, a))  # H_W Weight
        self.b_weight = nn.Parameter(torch.ones(2, 1, dim // 4, b))  # C_W Weight
        self.c_weight = nn.Parameter(torch.ones(2, dim // 4, c_h, c_w))  # C_H Weight
        self.dw_conv = InvertedDepthWiseConv2d(dim // 4, dim // 4)  # DW Conv

        # 创建三个神经网络层分别处理三个方向上的权重
        self.wg_a = nn.Sequential(  # H_W axis
            InvertedDepthWiseConv1d(dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv1d(2 * dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv1d(2 * dim // 4, dim // 4),
        )
        self.wg_b = nn.Sequential(  # C_W axis
            InvertedDepthWiseConv1d(dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv1d(2 * dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv1d(2 * dim // 4, dim // 4),
        )
        self.wg_c = nn.Sequential(  # C_H axis
            InvertedDepthWiseConv2d(dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv2d(2 * dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv2d(2 * dim // 4, dim // 4),
        )

        self.fusion_x1 = FeatureFusion()
        self.fusion_x2 = FeatureFusion()

    def forward(self, x):
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, c, a, b = x1.size()  # B C H W

        # ------- a convlution -------#
        x1 = x1.permute(0, 2, 1, 3)  # B H C W
        x1 = torch.fft.rfft2(x1, dim=(2, 3), norm='ortho')  # C_W H
        a_weight = self.a_weight
        a_weight = self.wg_a(F.interpolate(a_weight, size=x1.shape[2:4],
                                           mode='bilinear', align_corners=True
                                           ).squeeze(1)).unsqueeze(1).permute(1, 2, 3, 0)
        a_weight = torch.view_as_complex(a_weight.contiguous())
        x1 = x1 * a_weight
        x1 = torch.fft.irfft2(x1, s=(c, b), dim=(2, 3), norm='ortho').permute(0, 2, 1, 3)  # B C H W

        # ------- b convlution -------#
        x2 = x2.permute(0, 3, 1, 2)  # B W C H
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')  # C_H W
        b_weight = self.b_weight
        b_weight = self.wg_b(F.interpolate(b_weight, size=x2.shape[2:4],
                                           mode='bilinear', align_corners=True
                                           ).squeeze(1)).unsqueeze(1).permute(1, 2, 3, 0)
        b_weight = torch.view_as_complex(b_weight.contiguous())
        x2 = x2 * b_weight
        x2 = torch.fft.irfft2(x2, s=(c, a), dim=(2, 3), norm='ortho').permute(0, 2, 3, 1)  # B C H W

        # ------- c convlution -------## B C H W
        x3 = torch.fft.rfft2(x3, dim=(2, 3), norm='ortho')  # H_W C
        c_weight = self.c_weight
        c_weight = self.wg_c(F.interpolate(c_weight, size=x3.shape[2:4],
                                           mode='bilinear', align_corners=True
                                           )).permute(1, 2, 3, 0)
        c_weight = torch.view_as_complex(c_weight.contiguous())
        x3 = x3 * c_weight
        x3 = torch.fft.irfft2(x3, s=(a, b), dim=(2, 3), norm='ortho')

        # ------- Dw onvlution -------#
        x4 = self.dw_conv(x4)

        # ------- concat -------#
        output_x1 = self.fusion_x1(x1, x3)
        output_x2 = self.fusion_x2(x2, x3)

        out_m = torch.cat([output_x1, output_x2, x3, x4], dim=1)

        # ------- final output -------#
        out = out_m + x
        return out


class Conv1dGNGELU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.GroupNorm(4, out_channel),
            nn.GELU()
        )


class InvertedDepthWiseConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, expend_ratio=2):
        super().__init__()
        hidden_channel = in_channel * expend_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        # 1x1 pointwise conv
        layers.append(Conv1dGNGELU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # depthwise conv
            Conv1dGNGELU(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv
            nn.Conv1d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.GroupNorm(4, out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.mlp = nn.Sequential(InvertedDepthWiseConv2d(dim, mlp_ratio * dim),
                                 InvertedDepthWiseConv2d(mlp_ratio * dim, dim),
                                 nn.GELU()
                                 )

    def forward(self, x):
        return self.mlp(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.GroupNorm(4, dim)
        self.fn = fn

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class SFFM(nn.Module):
    def __init__(self, channel, depth, mlp_ratio=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(channel, FFSL(channel)),
                PreNorm(channel, SFAM(channel)),
                PreNorm(channel, MLP(channel, mlp_ratio)),
            ]))
    # all
    def forward(self, x):
        for attn, sfam, ff in self.layers:
            x_f = attn(x)
            x_s = sfam(x)
            x_F = x + x_f + x_s
            x = x + ff(x_F) + x_F

        return x

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, dilation=1, norm_type='gn', gn_num=4):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size, stride, padding=padding, dilation=dilation, groups=dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

        if norm_type == 'gn':
            self.norm_layer = nn.GroupNorm(gn_num, dim_out)
        elif norm_type == 'bn':
            self.norm_layer = nn.BatchNorm2d(dim_out)
        else:
            self.norm_layer = nn.BatchNorm2d(dim_out)

        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm_layer(x)
        return self.activation(x)


class Conv2dGNGELU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.GroupNorm(4, out_channel),
            nn.GELU()
        )


class InvertedDepthWiseConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, expend_ration=2):
        super().__init__()
        hidden_channel = in_channel * expend_ration
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        layers.append(Conv2dGNGELU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            Conv2dGNGELU(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=stride, groups=hidden_channel),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.GroupNorm(4, out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

class FeatureSelectionModule(nn.Module):
    def __init__(self, in_dim, reduction_ratio=4):
        super(FeatureSelectionModule, self).__init__()
        self.reduced_dim = in_dim // reduction_ratio
        self.reduction = nn.Sequential(
            nn.Conv2d(in_dim, self.reduced_dim, kernel_size=1, bias=False),
            nn.GroupNorm(4, self.reduced_dim),
            nn.GELU()
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.reduced_dim, self.reduced_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.reduced_dim // 2, self.reduced_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(self.reduced_dim, in_dim, kernel_size=1, bias=False),
            nn.GroupNorm(4, in_dim),
            nn.GELU()
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, encoder_features, decoder_features=None):
        if decoder_features is None:
            decoder_features = encoder_features
        reduced_features = self.reduction(encoder_features)
        channel_weights = self.channel_attention(reduced_features)
        channel_refined = reduced_features * channel_weights

        avg_pool = torch.mean(channel_refined, dim=1, keepdim=True)
        max_pool, _ = torch.max(channel_refined, dim=1, keepdim=True)
        spatial_features = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weights = self.spatial_attention(spatial_features)

        refined_features = channel_refined * spatial_weights
        output = self.output_conv(refined_features)

        return self.gamma * output + encoder_features

@MODELS.register_module()
class FSFDecodeHead(BaseDecodeHead):
    def __init__(self,
                 in_channels,
                 channels,
                 dim=[16, 32, 64, 128, 160],
                 depth=[1, 1, 1, 1],
                 mlp_ratio=4,
                 in_index=[0, 1, 2, 3, 4],  # Added this line
                 input_transform='multiple_select',
                 dropout_ratio=0.1,
                 align_corners=False,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 **kwargs):
        super(FSFDecodeHead, self).__init__(
            in_channels,
            channels,
            in_index=in_index,
            input_transform=input_transform,
            dropout_ratio=dropout_ratio,
            align_corners=align_corners,
            loss_decode=loss_decode,
            **kwargs)

        self.dim = dim
        self.d4 = nn.Sequential(
            SFFM(dim[4], depth[3], mlp_ratio),
            DepthWiseConv2d(dim[4], dim[3], kernel_size=3, padding=1, stride=1, dilation=1),
        )

        self.d3 = nn.Sequential(
            SFFM(dim[3], depth[2], mlp_ratio),
            DepthWiseConv2d(dim[3], dim[2], kernel_size=3, padding=1, stride=1, dilation=1),
        )

        self.d2 = nn.Sequential(
            SFFM(dim[2], depth[1], mlp_ratio),
            DepthWiseConv2d(dim[2], dim[1], kernel_size=3, padding=1, stride=1, dilation=1),
        )

        self.d1 = nn.Sequential(
            SFFM(dim[1], depth[0], mlp_ratio),
            DepthWiseConv2d(dim[1], dim[0], kernel_size=3, padding=1, stride=1, dilation=1),
        )

        self.d0 = nn.Sequential(
            nn.ConvTranspose2d(dim[0], dim[0], kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, dim[0]),
            nn.GELU(),
            nn.Conv2d(dim[0], self.num_classes, kernel_size=3, padding=1)
        )

        self.FSM_3 = FeatureSelectionModule(dim[3])
        self.FSM_2 = FeatureSelectionModule(dim[2])
        self.FSM_1 = FeatureSelectionModule(dim[1])
        self.FSM_0 = FeatureSelectionModule(dim[0])

    def _forward_feature(self, inputs):
        x = self._transform_inputs(inputs)
        x0, x1, x2, x3, x4 = x[0], x[1], x[2], x[3], x[4]  # 1/2, 1/4, 1/8, 1/16, 1/32

        out4 = F.interpolate(self.d4(x4), size=x3.shape[2:], mode='bilinear', align_corners=self.align_corners)
        out4 = self.FSM_3(x3, out4) + out4

        out3 = F.interpolate(self.d3(out4), size=x2.shape[2:], mode='bilinear', align_corners=self.align_corners)
        out3 = self.FSM_2(x2, out3) + out3

        out2 = F.interpolate(self.d2(out3), size=x1.shape[2:], mode='bilinear', align_corners=self.align_corners)
        out2 = self.FSM_1(x1, out2) + out2

        out1 = F.interpolate(self.d1(out2), size=x0.shape[2:], mode='bilinear', align_corners=self.align_corners)
        out1 = self.FSM_0(x0, out1) + out1

        return out1

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.d0(output)
        output = F.interpolate(output,
                               size=inputs[0].shape[2:],
                               mode='bilinear',
                               align_corners=self.align_corners)
        return output

@MODELS.register_module()
class FSFDecodeHead_T(FSFDecodeHead):
    def __init__(self, dim=[16, 32, 48, 96, 128], **kwargs):
        super().__init__(
            dim=dim,
            **kwargs
        )

@MODELS.register_module()
class FSFDecodeHead_S(FSFDecodeHead):
    def __init__(self, dim=[16, 32, 64, 128, 160], **kwargs):
        super().__init__(
            dim=dim,
            **kwargs
        )

@MODELS.register_module()
class FSFDecodeHead_B(FSFDecodeHead):
    def __init__(self, dim=[32, 48, 96, 128, 192], **kwargs):
        super().__init__(
            dim=dim,
            **kwargs
        )

@MODELS.register_module()
class FSFDecodeHead_L(FSFDecodeHead):
    def __init__(self, dim=[24, 32, 96, 320, 512], **kwargs):
        super().__init__(
            dim=dim,
            **kwargs
        )