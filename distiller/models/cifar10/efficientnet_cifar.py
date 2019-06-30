"""
 EfficientNet models in Distiller definition format.
 """
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

__all__ = ["efficientnetb0_cifar", "efficientnetb1_cifar",
           "efficientnetb2_cifar", "efficientnetb3_cifar"]


def get_efficientnet(which, pretrained=True):
    constructor = EfficientNet.from_pretrained if pretrained \
                    else EfficientNet.from_name
    model = constructor(f'efficientnet-{which}')
    model._fc = nn.Linear(in_features=1280, out_features=10)
    return model

def efficientnetb0_cifar(pretrained=False):
    return get_efficientnet("b0", pretrained)

def efficientnetb1_cifar(pretrained=False):
    return get_efficientnet("b1", pretrained)

def efficientnetb2_cifar(pretrained=False):
    return get_efficientnet("b2", pretrained)

def efficientnetb3_cifar(pretrained=False):
    return get_efficientnet("b3", pretrained)


if __name__=="__main__":
    efficientnetb0_cifar()