"""
 EfficientNet models in Distiller definition format.
 """
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

__all__ = ["efficientnetb0_cifar", "efficientnetb1_cifar",
           "efficientnetb2_cifar", "efficientnetb3_cifar"]


def get_efficientnet(which, pretrained=True):
    if pretrained:
        model = EfficientNet.from_pretrained(f'efficientnet-{which}')
        model._fc = nn.Linear(in_features=1280, out_features=10)
        return model
    else:
        raise NotImplementedError

def efficientnetb0_cifar():
    return get_efficientnet("b0", True)

def efficientnetb1_cifar():
    return get_efficientnet("b1", True)

def efficientnetb2_cifar():
    return get_efficientnet("b2", True)

def efficientnetb3_cifar():
    return get_efficientnet("b3", True)


if __name__=="__main__":
    efficientnetb0_cifar()