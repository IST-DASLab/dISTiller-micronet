EXP_NAME="effnet_imagenet_train_quantized"
CHECKPOINT_ROOT="/home/alex/checkpoints"
ARCH="efficientnetb1"
DATAPATH="/home/markovilya197/Datasets/imagenet"
CHECKPOINT="/home/alex/distiller-MicroNet/imagenet_efficientnet/checkpoints/effnet_imagenet_prune_base2_best.pth.tar"

LR=0.0001
MOM=0.9
WD=0.0

python compress_classifier.py \
	--name=$EXP_NAME \
	--arch=$ARCH $DATAPATH \
	--epochs=300 -p=30 -b=256 -j=8 \
	--lr=$LR --momentum=$MOM --wd=$WD \
	--pretrained \
	--vs=0 \
	--out-dir=$CHECKPOINT_ROOT \
	--resume-from=$CHECKPOINT \
	--gpus 0,1,2,3 \
	--reset-optimizer \
	--compress=./schedules/quantize_aware_training.yaml \
	--kd-teacher='efficientnetb2' \
	--kd-pretrained \
	--kd-temperature=2 \
	--kd-student-wt=0.3 \
	--kd-start-epoch=0 \