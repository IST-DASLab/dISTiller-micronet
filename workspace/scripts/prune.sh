# Pruning a pre-trained net

CHECKPOINT_ROOT="../../checkpoints"
# CHECKPOINT_MODEL="$CHECKPOINT_ROOT/effnet_train___2019.06.30-064721/effnet_train_best.pth.tar"
CHECKPOINT_MODEL="$CHECKPOINT_ROOT/effnet_train___2019.07.01-063516/effnet_train_checkpoint.pth.tar"

EXP_NAME="effnet_prune"

# python compress_classifier.py \
# 	--name=$EXP_NAME \
# 	--arch=efficientnetb0_cifar ../../data.cifar10 \
# 	--resume-from=$CHECKPOINT_MODEL \
# 	--epochs=20 -p=30 -j=1 \
# 	--lr=0.005 \
# 	--out-dir=$CHECKPOINT_ROOT \
# 	--compress=./schedules/efficientnetb0.schedule_agp.yaml \
# 	--reset-optimizer \
# 	--det --seed 42 # (if using these, set -j to 1)
# 	# --summary=sparsity - this will exit

python compress_classifier.py \
	--name=$EXP_NAME \
	--arch=efficientnetb0_cifar ../../data.cifar10 \
	--resume-from="../../checkpoints/effnet_prune___2019.07.01-071526/effnet_prune___2019.07.01-071526.log" \
	-p=30 -j=1 \
	--lr=0.005 \
	--out-dir=$CHECKPOINT_ROOT \
	--compress=./schedules/efficientnetb0.schedule_agp.yaml \
	--reset-optimizer \
	--evaluate \
	--det --seed 42 # (if using these, set -j to 1)
	# --summary=sparsity - this will exit
