"""
 EfficientNet models in Distiller definition format.
 """
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

__all__ = ["efficientnetb0_cifar100", "efficientnetb1_cifar100",
           "efficientnetb2_cifar100", "efficientnetb3_cifar100"]


def get_efficientnet(which, pretrained=True):
    constructor = EfficientNet.from_pretrained if pretrained \
                    else EfficientNet.from_name
    model = constructor(f'efficientnet-{which}')
    model._fc = nn.Linear(in_features=1280, out_features=100)
    return model

def efficientnetb0_cifar100(pretrained=False):
    return get_efficientnet("b0", pretrained)

def efficientnetb1_cifar100(pretrained=False):
    return get_efficientnet("b1", pretrained)

def efficientnetb2_cifar100(pretrained=False):
    return get_efficientnet("b2", pretrained)

def efficientnetb3_cifar100(pretrained=False):
    return get_efficientnet("b3", pretrained)


if __name__=="__main__":
    efficientnetb0_cifar100()