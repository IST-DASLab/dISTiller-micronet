import torch
import torch.nn as nn

import distiller.apputils as apputils
from distiller.models import create_model
from distiller.apputils import load_checkpoint



TEST_CHECKPOINT = '/nfs/scistore08/alistgrp/ashevche/distiller-data/checkpoints/effnet_imagenet_prune_base2\
___2019.07.07-231317/effnet_imagenet_prune_base2_checkpoint.pth.tar'

"""
TODO: Think if copies are needed if we start training from
scratch even if optimizer has momentum like property.
"""
class MaskedNet(nn.Module):
    def __init__(self, net, masks):
        super(MaskedNet, self).__init__()
        self.net = net

        self.copy, self.masks = dict(), dict()
        for name, param in self.net.named_parameters():
            if name in masks.keys():
                self.copy[name] = param.data.clone().detach()
                self.masks[name] = (param.data != 0).float()

    def forward(self, in_tensor):
        self.copy = dict()
        for name, param in self.net.named_parameters():
            if name in self.masks.keys():
                param.data *= self.masks[name]
        return self.net(in_tensor)

    def revert_pruned(self):
        for name, param in self.net.named_parameters():
            if name in self.masks.keys():
                mask = ~self.masks[name].byte()
                param.data[mask] = self.copy[name][mask]

def get_masked_model(checkpoint):
    model = create_model(False, 'imagenet', 'efficientnetb1')
    model, compression_scheduler, optimizer, start_epoch = apputils.load_checkpoint(model, checkpoint)
    masked_model = MaskedNet(model, compression_scheduler.zeros_mask_dict)
    return masked_model



if __name__ == '__main__':
    masked_net = get_masked_model(TEST_CHECKPOINT)
    in_tensor = torch.zeros(1, 3, 224, 224)
    masked_net(in_tensor)
    for name, param in masked_net.net.named_parameters():
        if name in masked_net.masks.keys():
            print((param == 0).sum())