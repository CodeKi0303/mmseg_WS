# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.decode_heads.psp_head import PPM

@HEADS.register_module()
class WS(BaseDecodeHead):
    def __init__(self, **kwargs):
        super(WS, self).__init__(input_transform='multiple_select', **kwargs)
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)
        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        self.weights = nn.Parameter(torch.ones(num_inputs, 1))
    
    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)
        out = 0
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            out += (
                torch.mul(self.weights[idx, :],
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)))
        out = self.cls_seg(out)
        return out