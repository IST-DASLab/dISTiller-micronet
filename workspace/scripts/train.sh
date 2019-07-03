# Call to pre-train before pruning

## Basic parameters, rarely need to change
# ARCH="efficientnetb0_cifar100"
ARCH="mobilenet_cifar"
DATAPATH="../../data.cifar10"
CHECKPOINT_ROOT="../../checkpoints/"

## The following arguments may be changed often
# this will be a prefix of dir within checkpoint dir:
# EXP_NAME="effnet_train"
EXP_NAME="mobilenet_train" #-- TODO

# optimizer parameters:
LR=0.1
MOM=0.9
WD=0.0005
# hopefully gets overriden by scheduler

python compress_classifier.py \
	--name=$EXP_NAME \
	--arch=$ARCH $DATAPATH \
	--epochs=200 -p 50 -j=1 \
	--lr=$LR --momentum=$MOM --wd=$WD \
	--out-dir=$CHECKPOINT_ROOT \
	--valid-size 0.05 \
	--noprint-weights-sparsity \
	--compress=./schedules/training.yaml \
        --det --seed 42 \
        --evaluate \
        --resume-from=/home/ks_korovina/checkpoints/mobilenet_train___2019.07.02-203747/mobilenet_train___2019.07.02-203747.log
        #--det --seed 42 # (if using these, set -j to 1)
        #--evaluate \
	#--pretrained \

# python compress_classifier.py \
# 	--name=$EXP_NAME \
# 	--arch=$ARCH $DATAPATH \
# 	--resume-from=$CHECKPOINT \
# 	--out-dir=$CHECKPOINT \
# 	--evaluate
