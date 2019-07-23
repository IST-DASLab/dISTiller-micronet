EXP_NAME="effnet_imagenet_prune_base2"
CHECKPOINT_ROOT="../../distiller-data/checkpoints"
ARCH="efficientnetb1"
DATAPATH="/home/imarkov/Datasets/imagenet"
CHECKPOINT="/nfs/scistore08/alistgrp/ashevche/distiller-data/checkpoints/effnet_imagenet_prune_base2___2019.07.05-172126/effnet_imagenet_prune_base2_checkpoint.pth.tar"

LR=0.01
MOM=0.9
WD=0.0

python compress_classifier.py \
	--name=$EXP_NAME \
	--arch=$ARCH $DATAPATH \
	--epochs=40 -p=30 -b=256 -j=30 \
	--lr=$LR --momentum=$MOM --wd=$WD \
	--out-dir=$CHECKPOINT_ROOT \
	--compress=./schedules/pruning2.yaml \
	--pretrained \
	--gpus=1,2,5,6,7 \
	# --kd-teacher "efficientnetb1" \
	# --kd-pretrained \
	# --kd-temperature=3.0 \
 #    --kd-student-wt=0.6 \
	# --kd-teacher-wt=0.0 \
	# --kd-start-epoch=0

# python compress_classifier.py \
# 	--name=$EXP_NAME \
# 	--arch=efficientnetb0_cifar ../../data.cifar10 \
# 	--resume-from="../../checkpoints/effnet_prune___2019.07.01-071526/effnet_prune___2019.07.01-071526.log" \
# 	-p=30 -j=1 \
# 	--lr=0.005 \
# 	--out-dir=$CHECKPOINT_ROOT \
# 	--compress=./schedules/efficientnetb0.schedule_agp.yaml \
# 	--reset-optimizer \
# 	--evaluate \
# 	--det --seed 42 # (if using these, set -j to 1)
# 	# --summary=sparsity - this will exit

# CHECKPOINT="../../checkpoints/resnet18__2019.07.02-222233/resnet18_best.pth.tar"    
# python compress_classifier.py \
# 	--name=$EXP_NAME \
# 	--arch=$ARCH $DATAPATH \
# 	--resume-from=$CHECKPOINT \
# 	--out-dir=$CHECKPOINT_ROOT \
# 	--gpus 5,6 \
	
# 	--evaluate