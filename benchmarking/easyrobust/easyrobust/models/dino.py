# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DINO: https://github.com/facebookresearch/dino
# --------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.registry import register_model


def load_pretrained_linear_weights(linear_classifier, model_name):
    url = None
    if model_name == "vits16":
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth"
    elif model_name == "vits8":
        url = "dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth"
    elif model_name == "vitb16":
        url = "dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth"
    elif model_name == "vitb8":
        url = "dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth"
    elif model_name == "resnet50":
        url = "dino_resnet50_pretrain/dino_resnet50_linearweights.pth"
    if url is not None:
        print("Load the reference pretrained linear weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)["state_dict"]
        state_dict['linear.bias'] = state_dict['module.linear.bias']
        state_dict['linear.weight'] = state_dict['module.linear.weight']
        del state_dict['module.linear.bias']
        del state_dict['module.linear.weight']
        linear_classifier.load_state_dict(state_dict, strict=True)
    else:
        print("Use random linear weights.")


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim: int, num_labels: int = 1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)
    

class DinoClassification(nn.Module):
    """ DINO classification with linear head
    """
    def __init__(self, model_name: str = 'vitb16', n_last_blocks: int = 1, avgpool_patchtokens: bool = True, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.n_last_blocks = n_last_blocks
        self.avgpool_patchtokens = avgpool_patchtokens
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('facebookresearch/dino:main', f'dino_{model_name}')
        if 'vit' in model_name:
            embed_dim = self.model.embed_dim * (n_last_blocks + int(avgpool_patchtokens))
        else:
            embed_dim = self.model.fc.weight.shape[1]
            self.model.fc = nn.Identity()
        self.linear_classifier = LinearClassifier(embed_dim, num_labels=1000)
        load_pretrained_linear_weights(self.linear_classifier, model_name)
        self.model.eval()
        self.linear_classifier.eval()
        self.model.to(self.device)
        self.linear_classifier.to(self.device)

    def forward(self, x):
        with torch.no_grad():
            if 'vit' in self.model_name:
                intermediate_output = self.model.get_intermediate_layers(x, self.n_last_blocks)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if self.avgpool_patchtokens:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = self.model(x)
        output = self.linear_classifier(output)
        return output


@register_model
def dino_vit_small_patch16(**kwargs):
    model = DinoClassification(model_name='vits16', n_last_blocks=4, avgpool_patchtokens=False, **kwargs)
    return model

@register_model
def dino_vit_small_patch8(**kwargs):
    model = DinoClassification(model_name='vits8', n_last_blocks=4, avgpool_patchtokens=False, **kwargs)
    return model

@register_model
def dino_vit_base_patch16(**kwargs):
    model = DinoClassification(model_name='vitb16', n_last_blocks=1, avgpool_patchtokens=True, **kwargs)
    return model

@register_model
def dino_vit_base_patch8(**kwargs):
    model = DinoClassification(model_name='vitb8', n_last_blocks=1, avgpool_patchtokens=True, **kwargs)
    return model

@register_model
def dino_resnet50(**kwargs):
    model = DinoClassification(model_name='resnet50', **kwargs)
    return model
    