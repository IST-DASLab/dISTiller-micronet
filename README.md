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
To use EfficientNet achitecture there are essentially two options. 

**Option 1.** Using manuall installation described here [here](https://github.com/lukemelas/EfficientNet-PyTorch).

**Option 2.** Using pip install:
```
$ pip install efficientnet_pytorch
```
As current pip version **has bugs** a hand-fix decribed in [commit](https://github.com/lukemelas/EfficientNet-PyTorch/commit/939d4abdeefc07e63d8bd42e7223365a4bc67942) is required.

### Troubleshooting

The following issues are system and setup dependent.

When trying to import pytorch on MacOSX, you may get `libomp.dylib can't be loaded` error. Then do `brew install libomp`.

If you get `AttributeError: 'os' module does not have 'sched_getaffinity' method`, which is likely to occur in MacOSX and some Linux distributions, try replacing the line with `os.cpu_count()`.

Also, do not forget `source setup.sh`.
