# Call to evaluate pretrained weights

EXP_NAME="effnet_imagenet_eval_initial_weights"
CHECKPOINT_ROOT="/nfs/scistore08/alistgrp/ashevche/distiller-data/checkpoints"
ARCH="efficientnetb1"
DATAPATH="/home/imarkov/Datasets/imagenet"
CHECKPOINT="/nfs/scistore08/alistgrp/ashevche/distiller-data/checkpoints/\
effnet_imagenet_prune_base2___2019.07.21-210409/effnet_imagenet_prune_base2_best.pth.tar"
 
python compress_classifier.py \
	--name=$EXP_NAME \
	--arch=$ARCH $DATAPATH \
	--pretrained \
	--resume-from=$CHECKPOINT \
	--gpus 0 \
	-j=5 \
	--evaluate \
	#--quantize-eval \
	#--qe-config-file=./schedules/quantize.yaml
	#--summary="sparsity"
