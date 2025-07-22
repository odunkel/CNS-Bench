import os
import sys
import torch
from trainscripts.textsliders.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV


sys.path.append(".")
sys.path.append("./trainscripts/textsliders")

def load_lora_network(args, unet):
    lora_weight = args.slider_weight
    if 'full' in lora_weight:
        train_method = 'full'
    elif 'noxattn' in lora_weight:
        train_method = 'noxattn'
    else:
        train_method = 'noxattn'

    print('train_method',train_method)

    network_type = "c3lier"
    if train_method == 'xattn':
        network_type = 'lierla'

    modules = DEFAULT_TARGET_REPLACE
    if network_type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV
    model_name = lora_weight

    name = os.path.basename(model_name)

    rank = args.rank
    alpha = 1.
    
    network = LoRANetwork(
            unet,
            rank=rank,
            multiplier=1.0,
            alpha=alpha,
            train_method=train_method,
        ).to(args.device, dtype=args.weight_dtype)
    network.load_state_dict(torch.load(lora_weight))
    return network