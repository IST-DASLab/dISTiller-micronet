# Call to evaluate pretrained weights

EXP_NAME="effnet_imagenet_eval"
CHECKPOINT_ROOT="/home/alex/checkpoints"
ARCH="efficientnetb1"
DATAPATH="/home/markovilya197/Datasets/imagenet"
CHECKPOINT="/home/alex/checkpoints/effnet_imagenet_train_quantized___2019.07.23-195312/effnet_imagenet_train_quantized_best.pth.tar"
 
python compress_classifier.py \
	--name=$EXP_NAME \
	--arch=$ARCH $DATAPATH \
	--pretrained \
	--resume-from=$CHECKPOINT \
	--gpus 2 \
	-j=5 \
	--evaluate \
	#--quantize-eval \
	# --qe-config-file=./schedules/quantize.yaml
	#--summary="sparsity"