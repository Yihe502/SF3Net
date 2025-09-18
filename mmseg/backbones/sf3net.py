import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torchvision.models._utils import IntermediateLayerGetter

from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, ConfigType

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

@MODELS.register_module()
class SF3Net(BaseModule):
    def __init__(self,
                 in_channels=3,
                 out_indices=(0, 1, 2, 3, 4),
                 dim=[16, 32, 48, 96, 128],
                 depth=[1, 1, 1, 1],
                 mlp_ratio=4,
                 pretrained='mmcls://mobilenet_v2',
                 init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.dim = dim
        self.depth = depth
        self.mlp_ratio = mlp_ratio

        mobilenet = mobilenet_v2(pretrained=pretrained)

        # 定义特征提取器
        self.backbone = IntermediateLayerGetter(
            mobilenet.features,
            return_layers={
                '1': 'stage0',
                '6': 'stage1',
                '13': 'stage2',
                '17': 'stage3',
                '18': 'stage4',
            }
        )

        self.adapter0 = DepthWiseConv2d(16, dim[0], kernel_size=1, stride=1, padding=0, dilation=1)
        self.adapter1 = DepthWiseConv2d(32, dim[1], kernel_size=1, stride=1, padding=0, dilation=1)
        self.adapter2 = DepthWiseConv2d(96, dim[2], kernel_size=1, stride=1, padding=0, dilation=1)
        self.adapter3 = DepthWiseConv2d(320, dim[3], kernel_size=1, stride=1, padding=0, dilation=1)
        self.adapter4 = DepthWiseConv2d(1280, dim[4], kernel_size=1, stride=1, padding=0, dilation=1)

        self.out_channels = dim

    def init_weights(self):
        if self._is_init:
            return
        if self.init_cfg is not None:
            super().init_weights()

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        features = self.backbone(x)

        x0 = self.adapter0(features["stage0"])  # 1/2
        x1 = self.adapter1(features["stage1"])  # 1/4
        x2 = self.adapter2(features["stage2"])  # 1/8
        x3 = self.adapter3(features["stage3"])  # 1/16
        x4 = self.adapter4(features["stage4"])  # 1/32

        outs = [x0, x1, x2, x3, x4]
        return [outs[i] for i in self.out_indices]

@MODELS.register_module()
class SF3Net_T(LSFFUNet):
    def __init__(self, dim=[16, 32, 48, 96, 128], **kwargs):
        super().__init__(
            dim=dim,
            depth=[4, 2, 2, 1],
            **kwargs
        )

@MODELS.register_module()
class SF3Net_S(LSFFUNet):
    def __init__(self, dim=[16, 32, 64, 128, 160], **kwargs):
        super().__init__(
            dim=dim,
            depth=[4, 2, 2, 1],
            **kwargs
        )

@MODELS.register_module()
class SF3Net_B(LSFFUNet):
    def __init__(self, dim=[32, 48, 96, 128, 192], **kwargs):
        super().__init__(
            dim=dim,
            depth=[4, 2, 2, 1],
            **kwargs
        )


@MODELS.register_module()
class SF3Net_L(LSFFUNet):
    def __init__(self, dim=[32, 48, 96, 320, 512], **kwargs):
        super().__init__(
            dim=dim,
            depth=[4, 2, 2, 1],
            **kwargs
        )
