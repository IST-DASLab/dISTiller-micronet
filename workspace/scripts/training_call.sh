# Call to pre-train before pruning

python compress_classifier.py \
	--arch efficientnetb0_cifar ../../data.cifar10 \
	-p 30 -j=2 \
	--lr=0.005 \
	--compress=./schedules/efficientnetb0.schedule_agp.yaml
