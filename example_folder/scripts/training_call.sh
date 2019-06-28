# Call to pre-train before pruning

python compress_classifier.py \
	--arch resnet20_cifar ../../data.cifar10 \
	-p 30 -j=2 \
	--lr=0.005 \
	--compress=./schedules/resnet18.schedule_agp.yaml
