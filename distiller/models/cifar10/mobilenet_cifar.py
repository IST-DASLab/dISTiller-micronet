"""MobileNet for CIFAR10"""

from distiller.models.imagenet import *
import torch.nn as nn

__all__=['mobilenet_cifar', 'mobilenet_025_cifar', 'mobilenet_050_cifar', 'mobilenet_075_cifar']

def preprocess_for_cifar10(model):
    model.fc = nn.Linear(model.channels[5], 10)
    return model

def mobilenet_025_cifar():
    model = mobilenet_025()
    return preprocess_for_cifar10(model)

def mobilenet_050_cifar():
    model = mobilenet_050()
    return preprocess_for_cifar10(model)

def mobilenet_075_cifar():
    model = mobilenet_075()
    return preprocess_for_cifar10(model)

def mobilenet_cifar():
    model = mobilenet()
    return preprocess_for_cifar10(model)
