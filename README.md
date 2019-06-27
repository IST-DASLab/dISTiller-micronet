# Distiller MicroNet
Forked [Distiller](https://github.com/NervanaSystems/distiller/tree/torch1.1-integration) repository for the NeurIPS 2019 MicroNet challenge

## Setup

First, clone the forked [distiller repository](https://github.com/alexturn/distiller-MicroNet).
```
$ git clone https://github.com/alexturn/distiller-MicroNet
```

If you want to use GPU support, ensure that your **CUDA.__version__==10** and install compatible PyTorch before distiller installation from the [official site](https://pytorch.org/get-started/locally/). After repository is cloned run 
```
$ cd distiller-MicroNet
$ pip install -e .
```

### EffecientNet
To use EfficientNet achitecture either follow the installation instructions from [here](https://github.com/lukemelas/EfficientNet-PyTorch) or install via pip:
```
$ pip install efficientnet_pytorch
```
**CURRENT PIP VERSION HAS BUGS USE MANUALL INSTALL** see fix in [commit](https://github.com/lukemelas/EfficientNet-PyTorch/commit/939d4abdeefc07e63d8bd42e7223365a4bc67942)
