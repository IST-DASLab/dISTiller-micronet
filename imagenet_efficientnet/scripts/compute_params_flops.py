"""
Script for computing submition metrics of efficientnetb1 model
"""

import torch
import distiller

from distiller.models import create_model



BEST_CKPT_PATH = '../checkpoints/effnet_imagenet_prune_base2_best.pth.tar'
DATASET_NAME = 'imagenet'
MODEL_NAME = 'efficientnetb1'

FULL_PRECISION_LAYERS = ['_conv_stem']
RATIO = 4

import effnet_flops

def _normalize_layer_name(layer_name):
    return layer_name.replace('module.', '')

def _normalize_distiller_state_dict(state_dict):
    return {_normalize_layer_name(k): v for k,v in state_dict.items()}

def load_ckpt():
    model = effnet_flops.EfficientNet.from_name('efficientnet-b1')
    if not torch.cuda.is_available():
        state_dict = torch.load(BEST_CKPT_PATH, map_location='cpu')
    else:
        state_dict = torch.load(BEST_CKPT_PATH)
    state_dict = _normalize_distiller_state_dict(state_dict['state_dict'])
    model.load_state_dict(state_dict)
    return model


def _is_not_quantized(name):
    return (
        'bias' in name or 
        any([name.startswith(prefix) for prefix in FULL_PRECISION_LAYERS])
    )

def _is_not_pruned(name):
    return 'bias' in name or 'bn' in name

def measure_effnet_storage(state_dict):
    total_used, total = 0. , 0.
    
    for name, param in state_dict.items():
        
        curr_ratio = RATIO
        if _is_not_quantized(name): 
            curr_ratio = 1.

        if '_fc.weight' in name:
            curr_ratio = 16. / 2.4

        total += param.numel()
        if _is_not_pruned(name):
            total_used += param.numel() / curr_ratio
        else:
            total_used += (param != 0).float().sum()  / curr_ratio
        
    return total_used.item(), total


def measure_effnet_flops(model):
    return model(torch.ones(1,3,224,224))



if __name__ == '__main__':
    model = load_ckpt()
    print(model)
    total_used_storage, total_storage = measure_effnet_storage(model.state_dict())
    print('Storage requirement for EfficientNET version b1:\n'
          f'\tTotal used storage: {total_used_storage:.0f},' 
          f' Total storage: {total_storage:.0f}',
          f' [Used ratio: {total_used_storage/total_storage:.6f}]')

    x, ops, total_ops = measure_effnet_flops(model)
    print('FLOPs measurements for EfficientNET version b1:\n'
          f'\tFLOPs: {ops:.0f},' 
          f' Vanila model FLOPs: {total_ops:.0f}',
          f' [Used ratio: {ops/total_ops:.6f}]')
