# Call to pre-train before pruning

## Basic parameters, rarely need to change
# ARCH="efficientnetb0_cifar100"
ARCH="wideresnet28x10_cifar"
DATAPATH="../../data.cifar10"
CHECKPOINT_ROOT="../../checkpoints/"

## The following arguments may be changed often
# this will be a prefix of dir within checkpoint dir:
EXP_NAME="wideresnet_train"

# optimizer parameters:
LR=0.1
MOM=0.9
WD=0.0005
# hopefully gets overriden by scheduler

python compress_classifier.py \
	--name=$EXP_NAME \
	--arch=$ARCH $DATAPATH \
	--epochs=180 -b 64 -p 50 -j=1 \
	--lr=$LR --momentum=$MOM --wd=$WD --nesterov \
	--out-dir=$CHECKPOINT_ROOT \
	--valid-size 0.05 \
	--noprint-weights-sparsity \
	--compress=./schedules/training.yaml \
    --det --seed 42 \
    --resume-from=../../checkpoints/wideresnet_train___2019.07.03-054148/wideresnet_train_checkpoint.pth.tar
    # --evaluate \
	# --pretrained \

# Now the 93.240 CIFAR10 model is in
# ../../checkpoints/wideresnet_train___2019.07.03-074217/wideresnet_train_checkpoint.pth.tar

# python compress_classifier.py \
# 	--name=$EXP_NAME \
# 	--arch=$ARCH $DATAPATH \
# 	--resume-from=$CHECKPOINT \
# 	--out-dir=$CHECKPOINT \
# 	--evaluate
