"""
 EfficientNet models in Distiller definition format.
 """
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

__all__ = ["efficientnet_cifar_b0", "efficientnet_cifar_b1",
        "efficientnet_cifar_b2", "efficientnet_cifar_b3"]


def get_efficientnet(which, pretrained=True):
    if pretrained:
        model = EfficientNet.from_pretrained(f'efficientnet-{which}')
        model._fc = nn.Linear(in_features=1280, out_features=10)
        return model
    else:
        raise NotImplementedError

def efficientnet_cifar_b0():
    return get_efficientnet("b0", True)

def efficientnet_cifar_b1():
    return get_efficientnet("b1", True)

def efficientnet_cifar_b2():
    return get_efficientnet("b2", True)

def efficientnet_cifar_b3():
    return get_efficientnet("b3", True)


if __name__=="__main__":
    efficientnet_cifar_b0()