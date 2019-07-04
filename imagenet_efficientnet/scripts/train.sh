# Call to pre-train before pruning

EXP_NAME="effnet_imagenet_prune"
CHECKPOINT_ROOT="/nfs/scistore08/alistgrp/ashevche/checkpoints"

ARCH="efficientnetb0_imagenet"
DATAPATH="../../data.cifar10"
CHECKPOINT_ROOT="../../checkpoints/"

## The following arguments may be changed often
# this will be a prefix of dir within checkpoint dir:
EXP_NAME="resnet18"
# EXP_NAME="mobilenet_train" -- TODO

# optimizer parameters:
LR=0.1
MOM=0.9
WD=0.0005  # same as default
# hopefully gets overriden by scheduler

python compress_classifier.py \
	--name=$EXP_NAME \
	--arch=$ARCH $DATAPATH \
	--epochs=180 -p 50 -j=4 \
	--lr=$LR --momentum=$MOM --wd=$WD \
	--out-dir=$CHECKPOINT_ROOT \
	--valid-size 0.05 \
	--noprint-weights-sparsity \
	--compress=./schedules/training.yaml \
	--gpus 0,1,2,3 \
	--det --seed 42 # (if using these, set -j to 1)


# CHECKPOINT="../../checkpoints/resnet18__2019.07.02-222233/resnet18_best.pth.tar"    
# python compress_classifier.py \
# 	--name=$EXP_NAME \
# 	--arch=$ARCH $DATAPATH \
# 	--resume-from=$CHECKPOINT \
# 	--out-dir=$CHECKPOINT_ROOT \
# 	--gpus 5,6 \
	
# 	--evaluate