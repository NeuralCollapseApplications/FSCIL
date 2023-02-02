# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch.nn as nn
from mmcv.cnn import build_norm_layer

from mmcls.models.builder import NECKS


@NECKS.register_module()
class MLPFFNNeck(nn.Module):
    def __init__(self, in_channels=512, out_channels=512):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.ln1 = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_channels, in_channels * 2)),
            ('ln', build_norm_layer(dict(type='LN'), in_channels * 2)[1]),
            ('relu', nn.LeakyReLU(0.1))
        ]))
        self.ln2 = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_channels * 2, in_channels * 2)),
            ('ln', build_norm_layer(dict(type='LN'), in_channels * 2)[1]),
            ('relu', nn.LeakyReLU(0.1))
        ]))
        self.ln3 = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_channels * 2, out_channels, bias=False)),
        ]))
        if in_channels == out_channels:
            # self.ffn = nn.Identity()
            self.ffn = nn.Sequential(OrderedDict([
                ('proj', nn.Linear(in_channels, out_channels, bias=False)),
            ]))
        else:
            self.ffn = nn.Sequential(OrderedDict([
                ('proj', nn.Linear(in_channels, out_channels, bias=False)),
            ]))

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[-1]
        x = self.avg(inputs)
        x = x.view(inputs.size(0), -1)
        identity = x
        x = self.ln1(x)
        x = self.ln2(x)
        x = self.ln3(x)
        x = x + self.ffn(identity)
        return x
