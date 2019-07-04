"""
 EfficientNet models in Distiller definition format.
 """
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

__all__ = ["efficientnetb0", "efficientnetb1",
           "efficientnetb2", "efficientnetb3"]


def get_efficientnet(which, pretrained=True):
    constructor = EfficientNet.from_pretrained if pretrained \
                    else EfficientNet.from_name
    model = constructor(f'efficientnet-{which}')
    return model

def efficientnetb0(pretrained=False):
    return get_efficientnet("b0", pretrained)

def efficientnetb1(pretrained=False):
    return get_efficientnet("b1", pretrained)

def efficientnetb2(pretrained=False):
    return get_efficientnet("b2", pretrained)

def efficientnetb3(pretrained=False):
    return get_efficientnet("b3", pretrained)


if __name__=="__main__":
    efficientnetb0()