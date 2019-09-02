"""
An new script for computing parameter storage and FLOPs.
Updated from the previous script using the example from micronet orgs:
https://github.com/google-research/google-research/blob/master/micronet_challenge/counting.py
Implementation of EfficientNet they are using:
https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py

Notes:
* Output is (should be? check with orgs?) in MBytes of parameter storage, including bit masks.
* Batchnorms: we don't need to count them blindly; actually, at serving time, we can merge them into convs,
    and hence we don't waste extra storage for bn scales, but still need to count biases if convs don't have them.
    To not double count biases, we check of for every weight there is a bias also in the names dict.
    This works for the current EfficientNet implementation, because the only layers that have weight,
    but not bias, are conv-type layers, and they match up 1-to-1 with batchnorm operations.
"""

import torch
import torch.nn as nn
import numpy as np

import distiller
from distiller.models import create_model
import effnet_flops

BEST_CKPT_PATH = '../checkpoints/effnet_imagenet_prune_base2_best.pth.tar'
DATASET_NAME = 'imagenet'
MODEL_NAME = 'efficientnetb1'
FULL_PRECISION_LAYERS = ['_conv_stem']


def load_ckpt():
    def _normalize_layer_name(layer_name):
        return layer_name.replace('module.', '')
    def _normalize_distiller_state_dict(state_dict):
        return {_normalize_layer_name(k): v for k,v in state_dict.items()}
    model = effnet_flops.EfficientNet.from_name('efficientnet-b1')
    if not torch.cuda.is_available():
        state_dict = torch.load(BEST_CKPT_PATH, map_location='cpu')
    else:
        state_dict = torch.load(BEST_CKPT_PATH)
    state_dict = _normalize_distiller_state_dict(state_dict['state_dict'])
    model.load_state_dict(state_dict)
    return model


def is_not_quantized(name):
    return (
        'bias' in name or 
        any([name.startswith(prefix) for prefix in FULL_PRECISION_LAYERS])
    )

def get_sparse_size(tensor_shape, param_bits, sparsity):
  """Given a tensor shape returns #bits required to store the tensor sparse.
  If sparsity is greater than 0, we do have to store a bit mask to represent
  sparsity.
  Args:
    tensor_shape: list<int>, shape of the tensor
    param_bits: int, number of bits the elements of the tensor represented in.
    sparsity: float, sparsity level. 0 means dense.
  Returns:
    int, number of bits required to represented the tensor in sparse format.
  """
  n_elements = np.prod(tensor_shape)
  c_size = n_elements * param_bits * (1 - sparsity)
  if sparsity > 0:
    c_size += n_elements  # 1 bit binary mask
  return c_size


def compute_effnet_param_storage(state_dict, max_bits=16):
    total_bits_used, total_bits = 0., 0.
    for name, param in state_dict.items():
        if 'bn' in name:
            # They don't count batchnorms separately for serving time,
            # and merge them into the convs instead (no extra params for scales),
            # but need to count biases storage if conv2d doesn't have it.
            # See the check at the end of the loop body for this.
            continue
        if is_not_quantized(name):
            # biases and first conv are not quantized at all
            param_bits = max_bits
        elif '_fc.weight' in name:
            # last linear layer is quantized to 2.5
            param_bits = 2.5
        else:
            # all other cases
            param_bits = 4

        tensor_shape = param.shape
        sparsity = (param == 0).float().mean()
        total_bits_used += get_sparse_size(tensor_shape, param_bits, sparsity)
        total_bits += get_sparse_size(tensor_shape, max_bits, sparsity=0.)

        # We can merge the batchnorm scales into other params. But biases need to be stored if not already.
        # NOTE: The code patch below is specific to MBConvBlock implementation, as it assumes batchnorms are 
        # matched up 1:1 with one of some op that has weight, but no bias (which is conv).
        if name.endswith('weight') and name.replace('weight', 'bias') not in state_dict:
            total_bits_used += get_sparse_size([tensor_shape[0]], param_bits, sparsity)
            total_bits += get_sparse_size([tensor_shape[0]], max_bits, sparsity=0.)
    return total_bits_used, total_bits


def compute_effnet_flops(model):
    # Running forward on a model that computes all FLOPs along the way
    # Just one example is passed (batch size = 1).
    dummy_example = torch.ones(1, 3, 224, 224)
    return model(dummy_example)


def get_final_scores(storage_bits, num_math_ops):
    """
    Compute scores as fraction of MobileNetV2 requirements.
    The reference model has 6.9M parameters and 1170M math operations.

    Arguments:
        storage_bits -- number of bits in storage of the model.
            As we are performing quantization, 32 bits=1 parameter,
            hence we can compute the number of parameters as
            storage_bits / 32.
        num_math_ops -- number of summation and multiplication operations.
            This already accounts for quantization that we performed:
            e.g. a 4-bit operation (performed between a 4-bit weight and 4-bit input)
            counts as 4 / 32 of an operation.
    Return:
        score_storage, score_math_ops, total_score

    """
    BASELINE_STORAGE_PARAMS = 6.9e6
    BASELINE_NUM_MATH_OPS = 1170e6
    score_storage = storage_bits / 32 / BASELINE_STORAGE_PARAMS
    score_math_ops = num_math_ops / BASELINE_NUM_MATH_OPS
    return score_storage, score_math_ops, score_storage + score_math_ops


if __name__ == "__main__":
    model = load_ckpt()
    total_bits_used, total_bits = compute_effnet_param_storage(model.state_dict())

    print('\n=== 1. Model statistics =============================================================================')

    print('Storage requirement for EfficientNet version b1:\n'
        f'\tTotal storage of sparsified and quantized model: {total_bits_used/8/1e6:.4f} MBytes ({total_bits_used/32:.0f} params)\n'
        f'\tTotal storage of original model: {total_bits/8/1e6:.4f} MBytes\n'
        f'\tRelative improvement over original model {total_bits_used/total_bits:.4f}'
    )

    x, ops, full_ops = compute_effnet_flops(model)
    print('EfficientNet version b-1 FLOPS requirement:\n'
        f'\tTotal MFLOPs of sparsified and quantized model {ops/1e6:.4f}\n',
        f'\tTotal MFLOPs original model {full_ops/1e6:.4f}\n',
        f'\tRelative improvement over original model {ops/full_ops:.4f}'
    )

    print('\n=== 2. Model result scores (relative to MobileNetV2) =================================================')
    score_storage, score_math_ops, total_score = get_final_scores(total_bits_used, ops)
    print(
        f'Storage score: {score_storage:.4f}\n'
        f'Math ops score: {score_math_ops:.4f}\n'
        f'Overall score: {total_score:.4f}'
    )

