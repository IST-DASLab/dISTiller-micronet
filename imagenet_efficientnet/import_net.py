import torch
import torch.nn as nn

import distiller.apputils as apputils
from distiller.models import create_model
from distiller.apputils import load_checkpoint


__all__ = ['get_masked_model_from_distiller_checkpoint',
           'get_masked_model_from_quantdistil_checkpoint']

TEST_CHECKPOINT = '/nfs/scistore08/alistgrp/ashevche/distiller-data/checkpoints/effnet_imagenet_prune_base2\
___2019.07.07-231317/effnet_imagenet_prune_base2_checkpoint.pth.tar'


class MaskedNet(nn.Module):
    def __init__(self, net, maskers=None):
        super(MaskedNet, self).__init__()
        self.net = net
        self.hooks = None
        self.masks = None

        if maskers is not None:
            self.masks = {name: maskers[name].mask for name in maskers.keys()}
            self.register_hooks()

    def forward(self, in_tensor):
        return self.net(in_tensor)

    def register_hooks(self):
        if self.hooks is not None:
            raise ValueError('Please remove previous hooks before new created')
        if self.masks is None:
            raise ValueError('The are no masks provided in the __init__')

        self.hooks = []
        for name, param in self.net.named_parameters():
            self.hooks.append(
                param.register_hook(self._build_masking_hook(self.masks[name]))
            )

    def remove_hooks(self):
        if self.hooks is None:
            raise ValueError('No hooks to remove')
        for hook in self.hooks:
            hook.remove()

    @staticmethod
    def _build_masking_hook(mask):
        hook = lambda grad: grad * mask
        return hook

def get_masked_model_from_distiller_checkpoint(checkpoint):
    model = create_model(False, 'imagenet', 'efficientnetb1')
    model, compression_scheduler, optimizer, start_epoch = apputils.load_checkpoint(model, checkpoint)
    masked_model = MaskedNet(model, compression_scheduler.zeros_mask_dict)
    return masked_model

def get_masked_model_from_quantdistil_checkpoint(model, distiller_checkpoint=None):
    maskers = None
    if distiller_checkpoint is not None:
        maskers = apputils.load_checkpoint(model, checkpoint)[1].zeros_mask_dict
    masked_model = MaskedNet(model, maskers)
    return masked_model


if __name__ == '__main__':
    masked_net = get_masked_model_from_distiller_checpoint(TEST_CHECKPOINT)
    print(masked_net.state_dict())
