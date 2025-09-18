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
    '''
*****
    '''

class FFSL(nn.Module):
    '''
*****
    '''

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
    '''
*****
    '''


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
class SF3DecodeHead(BaseDecodeHead):
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
class SF3DecodeHead_T(FSFDecodeHead):
    def __init__(self, dim=[16, 32, 48, 96, 128], **kwargs):
        super().__init__(
            dim=dim,
            **kwargs
        )

@MODELS.register_module()
class SF3DecodeHead_S(FSFDecodeHead):
    def __init__(self, dim=[16, 32, 64, 128, 160], **kwargs):
        super().__init__(
            dim=dim,
            **kwargs
        )

@MODELS.register_module()
class SF3DecodeHead_B(FSFDecodeHead):
    def __init__(self, dim=[32, 48, 96, 128, 192], **kwargs):
        super().__init__(
            dim=dim,
            **kwargs
        )

@MODELS.register_module()
class SF3DecodeHead_L(FSFDecodeHead):
    def __init__(self, dim=[24, 32, 96, 320, 512], **kwargs):
        super().__init__(
            dim=dim,
            **kwargs
        )
