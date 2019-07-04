# Analyze sensitivity of a given model using batches of data.
# Requires a trained model or a model checkpoint.

# Load a model only to create a checkpoint
ARCH="efficientnetb0"
DATAPATH="../../data.imagenet"
CHECKPOINT_ROOT="../../checkpoints/"

# this will be a prefix of dir within checkpoint dir:
EXP_NAME="efficientnet_sensitivity"
CHECKPOINT_PATH=$CHECKPOINT_ROOT/"efficientnet_train_imagenet___2019.07.04-043141/efficientnet_train_imagenet_checkpoint.pth.tar"

# optimizer parameters:
LR=0.1
MOM=0.9
WD=0.0005

python compress_classifier.py \
    --name=$EXP_NAME \
    --arch=$ARCH $DATAPATH \
    --epochs=0 -b 64 -p 50 -j=1 \
    --lr=$LR --momentum=$MOM --wd=$WD --nesterov \
    --out-dir=$CHECKPOINT_ROOT \
    --valid-size 0.05 \
    --noprint-weights-sparsity \
    --compress=./schedules/training.yaml \
    --det --seed 42 \

# Use that checkpoint to perform sensitivity analysis
python3 compress_classifier.py \
    -a $ARCH $DATAPATH \
    -j 4 \
    --resume=$CHECKPOINT_PATH \
    --sense=filter
