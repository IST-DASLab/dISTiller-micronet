# Call to pre-train before pruning

## Basic parameters, rarely need to change
ARCH="efficientnetb0_cifar"
DATAPATH="../../data.cifar10"
CHECKPOINT="../../checkpoints/"

## The following arguments may be changed often
# this will be a prefix of dir within checkpoint dir:
EXP_NAME="effnet_train"

# optimizer parameters:
LR=1e-1
MOM=0.9
WD=1e-4  # same as default


python compress_classifier.py \
	--name=$EXP_NAME \
	--arch=$ARCH $DATAPATH \
	--epochs=70 -p 30 -j=2 \
	--lr=$LR --momentum=$MOM --wd=$WD \
	# --pretrained \
	--out-dir=$CHECKPOINT \
	--valid-size 0.05 \
	--noprint-weights-sparsity \
	--compress=./schedules/training.yaml
	# -det --seed 42 (if using these, set -j to 1)

# python compress_classifier.py \
# 	--name=$EXP_NAME \
# 	--arch=$ARCH $DATAPATH \
# 	--resume-from=$CHECKPOINT \
# 	--out-dir=$CHECKPOINT \
# 	--evaluate
