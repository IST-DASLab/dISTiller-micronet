# Call to pre-train before pruning

## Basic parameters, rarely need to change
ARCH="efficientnetb0_cifar100"
# ARCH="mobilenet_cifar"
DATAPATH="../../data.cifar100"
CHECKPOINT_ROOT="../../checkpoints/"

## The following arguments may be changed often
# this will be a prefix of dir within checkpoint dir:
EXP_NAME="effnet_train"
# EXP_NAME="mobilenet_train" -- TODO

# optimizer parameters:
LR=0.1
MOM=0.9
WD=0.  # same as default
# hopefully gets overriden by scheduler

python compress_classifier.py \
	--name=$EXP_NAME \
	--arch=$ARCH $DATAPATH \
	--epochs=150 -p 30 -j=1 \
	--lr=$LR --momentum=$MOM --wd=$WD \
	--out-dir=$CHECKPOINT_ROOT \
	--valid-size 0.05 \
	--noprint-weights-sparsity \
	--compress=./schedules/training.yaml \
	--pretrained \
	--det --seed 42 # (if using these, set -j to 1)

# python compress_classifier.py \
# 	--name=$EXP_NAME \
# 	--arch=$ARCH $DATAPATH \
# 	--resume-from=$CHECKPOINT \
# 	--out-dir=$CHECKPOINT \
# 	--evaluate
