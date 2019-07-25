# Distiller MicroNet
We have forked the [Distiller](https://github.com/NervanaSystems/distiller/tree/torch1.1-integration) repository for the NeurIPS 2019 MicroNet challenge

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

**Google Cloud.** If you are using Google cloud or otherwise want to start installation from scratch, the [Cookbook](https://github.com/alexturn/distiller-MicroNet/blob/master/workspace/Cookbook.md) may be useful.

**Final checkpoint.** Download final model [checkpoint](https://www.dropbox.com/s/ce2kwplnyc3jdl9/effnet_imagenet_train_quantized_best.pth.tar?dl=0) file `effnet_imagenet_train_quantized_best.pth.tar`. After that move it
in folder `distiller-MicroNet/imagenet_efficientnet/checkpoints`.

### EfficientNet
To use EfficientNet achitecture there are essentially two options. 

**Option 1.** Using manual installation described here [here](https://github.com/lukemelas/EfficientNet-PyTorch).

**Option 2.** Using pip install:
```
$ pip install efficientnet_pytorch
```
As current pip version **has bugs** a hand-fix decribed in [commit](https://github.com/lukemelas/EfficientNet-PyTorch/commit/939d4abdeefc07e63d8bd42e7223365a4bc67942) is required. (**UPDATE:** the pip version is fixed now.)

### Troubleshooting

The following issues are system and setup dependent.

When trying to import pytorch on MacOSX, you may get `libomp.dylib can't be loaded` error. Then do `brew install libomp`.

If you get `AttributeError: 'os' module does not have 'sched_getaffinity' method`, which is likely to occur in MacOSX and some Linux distributions, try replacing the line with `os.cpu_count()`.

Also, do not forget `source setup.sh`.

## Approach

In general, the approach follows two steps. 

First, we prune the model layer-wise with desired target sparsity levels (specified in `imagenet_efficientnet/schedules/pruning2.yaml`) using
Automated Gradual Pruner. 

After that, we quantize the weights and activations to 4 BITS and use quantized aware training procedure with knowledge distillation to fine-tune the model (specified in `imagenet_efficientnet/schedules/quantize_aware_training.yaml` with bits for layers) for 1 epoch.

**Notes:** We do not quantize the bias term and first convolution (`_conv_stem` layer of PyTorch model) including its activations, the last linear layer weight is quantized to a fractional number of BITs 2.5 the activations of this
layer are not quantized.

### Results

The results of final model are listed in the table below. For details on competition metrics evaluation see the corresponding section.

| Metric       | Our Model      | Vanila model  |  Ratio  |
|    :---:     |     :---:      |     :---:     |  :---:  |
| Storage      | 773464     	| 7856301       | 0.0985  |
| FLOPs        | 77565880       | 544357991     | 0.1425  |

**Final relative MobileNet-V2 score:** `773464 / 6.9M + 77565880 / 1170 M` that is approximately **0.178392**


## Reproducing the checkpoints

Go to the `imagenet_efficientnet/` folder. The first step of the whole procedure is pruning. To invoke pruning run
```
$ bash scripts/prune.sh
```
under the `imagenet_efficientnet/` directory. You might need to modify the GPU_IDS (`--gpus`), number of workers (`-j`), batch size (`-b`) and path to ImageNet (`DATAPATH`) according to your machine configs.
Probably you will need to modify experiment name (`EXP_NAME`) and path to the chekpoint dir (`CHECKPOINT_ROOT`) to be more suitable for you.

**Note:** the checkpoint for this step is available at `imagenet_efficientnet/checkpoints` dir named `effnet_imagenet_prune_base2_best.pth.tar`. For evaluation of checkpoints see corresponding section.

After that, quantization follows. It can be invoked with command
```
$ bash scripts/train_quantized.sh
```
Please note, that as before you should change some values according to your machine config.


## Evaluation of checkpoints

To evaluate the checkpoint model Top-1 on ImageNet you should modify checkpoint path ('CHECKPOINT') in `scripts/eval.sh` accordingly.
For final model evaluation modify checkpoint path as follows 

```CHECKPOINT=$ABSOLUTE_PATH_PREFIX/distiller-MicroNet/imagenet_efficientnet/checkpoints/effnet_imagenet_train_quantized_best.pth.tar```

After these modifications, invoke Top-1 evaluation running:
```
$ bash scripts/eval.sh
```

## Competition metrics (storage and flops)

We accompany our submission with the evaluation script to compute storage requirements and number of flops. For each metric we have three values: the corresponding metric on final model,
metric of vanila (i.e. model before pruning and quantization) and the ratio of both. 

To invoke the metrics script run
```
$ python compute_params_flops.py
```
under the `scripts` folder. The script is rather simple and is fully contained in `compute_params_flops.py`. For flops we modify the model forward and upload the weights to compute resulting metric.
This pipeline is implemented in `scripts/effnet_flops.py`. We consider residual connections, activations and batch norm flops among others in this procedure.

## Contact

If you have any questions regarding evaluation, reproduction step or the implementation of competition metrics, feel free to contact us on: `alexmtcore@gmail.com`.



