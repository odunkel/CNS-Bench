# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DINOv2: https://github.com/facebookresearch/dinov2
# --------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.registry import register_model


class DinoV2Classification(nn.Module):
    """ DINOv2 classification with linear head
    """
    def __init__(self, model_name: str = "vitb14", with_registers: bool = True, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.with_registers = with_registers
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        load_str = f'dinov2_{model_name}'
        if with_registers:
            load_str += '_reg'
        load_str += '_lc'
        self.model = torch.hub.load('facebookresearch/dinov2', load_str)
        self.model.eval()
        self.model.to(self.device)

    def forward(self, x):
        with torch.no_grad():
            output = self.model(x)
        return output


@register_model
def dinov2_vit_small_patch14(**kwargs):
    model = DinoV2Classification(model_name='vits14', with_registers=False, **kwargs)
    return model

@register_model
def dinov2_vit_small_patch14_reg(**kwargs):
    model = DinoV2Classification(model_name='vits14', with_registers=True, **kwargs)
    return model

@register_model
def dinov2_vit_base_patch14(**kwargs):
    model = DinoV2Classification(model_name='vitb14', with_registers=False, **kwargs)
    return model

@register_model
def dinov2_vit_base_patch14_reg(**kwargs):
    model = DinoV2Classification(model_name='vitb14', with_registers=True, **kwargs)
    return model

@register_model
def dinov2_vit_large_patch14(**kwargs):
    model = DinoV2Classification(model_name='vitl14', with_registers=False, **kwargs)
    return model

@register_model
def dinov2_vit_large_patch14_reg(**kwargs):
    model = DinoV2Classification(model_name='vitl14', with_registers=True, **kwargs)
    return model

@register_model
def dinov2_vit_giant_patch14(**kwargs):
    model = DinoV2Classification(model_name='vitg14', with_registers=False, **kwargs)
    return model

@register_model
def dinov2_vit_giant_patch14_reg(**kwargs):
    model = DinoV2Classification(model_name='vitg14', with_registers=True, **kwargs)
    return model
