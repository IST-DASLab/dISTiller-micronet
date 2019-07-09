import torch
import torch.nn as nn

import distiller.apputils as apputils
from distiller.models import create_model
from distiller.apputils import load_checkpoint


__all__ = ['get_masked_model']

TEST_CHECKPOINT = '/nfs/scistore08/alistgrp/ashevche/distiller-data/checkpoints/effnet_imagenet_prune_base2\
___2019.07.07-231317/effnet_imagenet_prune_base2_checkpoint.pth.tar'


class MaskedNet(nn.Module):
    def __init__(self, net, masks):
        super(MaskedNet, self).__init__()
        self.net = net
        self.masks = dict()

        for name, param in self.net.named_parameters():
            if name in masks.keys():
                self.masks[name] = (param.data != 0).float()
                param.register_hook(self._build_masking_hook(self.masks[name]))

    def forward(self, in_tensor):
        return self.net(in_tensor)

    @staticmethod
    def _build_masking_hook(mask):
        hook = lambda grad: grad * mask
        return hook

def get_masked_model(checkpoint):
    model = create_model(False, 'imagenet', 'efficientnetb1')
    model, compression_scheduler, optimizer, start_epoch = apputils.load_checkpoint(model, checkpoint)
    masked_model = MaskedNet(model, compression_scheduler.zeros_mask_dict)
    return masked_model


if __name__ == '__main__':
    masked_net = get_masked_model(TEST_CHECKPOINT)
    in_tensor = torch.ones(1, 3, 224, 224)
    loss = nn.CrossEntropyLoss()(masked_net(in_tensor), torch.ones(1).long().cuda())
    loss.backward()
    for name, param in masked_net.net.named_parameters():
        if name in masked_net.masks.keys():
            print((param.grad.data == 0).sum())