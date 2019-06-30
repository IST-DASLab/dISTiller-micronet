# Pruning a pre-trained net

CHECKPOINT="../../checkpoints/"

python compress_classifier.py \
	--arch=efficientnetb0_cifar ../../data.cifar10 \
	--resume-from=$CHECKPOINT \
	-p=30 -j=2 \
	--lr=0.005 \
	--compress=./schedules/efficientnetb0.schedule_agp.yaml
	# --summary=sparsity - this will exit
