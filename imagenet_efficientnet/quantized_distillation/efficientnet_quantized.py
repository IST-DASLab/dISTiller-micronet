import os
import torch
import torchvision
import cnn_models.conv_forward_model as convForwModel
import cnn_models.help_fun as cnn_hf
import datasets
import model_manager

import distiller
from distiller.apputils import imagenet_get_datasets
from import_net import *

cuda_devices = [3,4,5,6,7]

if 'NUM_BITS' in os.environ:
    NUM_BITS = int(os.environ['NUM_BITS'])
else:
    NUM_BITS = 8

print('Number of bits in training: {}'.format(NUM_BITS))

TEST_CHECKPOINT = '/nfs/scistore08/alistgrp/ashevche/distiller-data/checkpoints/effnet_imagenet_prune_base2\
___2019.07.07-231317/effnet_imagenet_prune_base2_checkpoint.pth.tar'

datasets.BASE_DATA_FOLDER = '...'
SAVED_MODELS_FOLDER = '...'

USE_CUDA = torch.cuda.is_available()
NUM_GPUS = len(cuda_devices)

try:
    os.mkdir(datasets.BASE_DATA_FOLDER)
except:pass
try:
    os.mkdir(SAVED_MODELS_FOLDER)
except:pass

epochsToTrainImageNet = 20
imageNetmodelsFolder = os.path.join(SAVED_MODELS_FOLDER, 'imagenet_new')
imagenet_manager = model_manager.ModelManager('model_manager_imagenet_distilled_New{}bits.tst'.format(NUM_BITS),
                                              'model_manager', create_new_model_manager=False)

for x in imagenet_manager.list_models():
    if imagenet_manager.get_num_training_runs(x) >= 1:
        s = '{}; Last prediction acc: {}, Best prediction acc: {}'.format(x,
                                            imagenet_manager.load_metadata(x)[1]['predictionAccuracy'][-1],
                                            max(imagenet_manager.load_metadata(x)[1]['predictionAccuracy']))
        print(s)

try:
    os.mkdir(imageNetmodelsFolder)
except:pass

train_dset, test_dset = imagenet_get_datasets("../../../distiller-data/data.imagenet")

train_loader = torch.utils.data.DataLoader(train_dset, batch_size=256, 
                                           num_workers=30, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1000, 
                                           num_workers=10, pin_memory=True)


bits_to_try = [NUM_BITS]
#efficientnetb2 = distiller.models.create_model(True, 'imagenet', 'efficientnetb2')

for numBit in bits_to_try:
    efficientnetb1 = get_masked_model_from_distiller_checkpoint(TEST_CHECKPOINT)
    #if NUM_GPUS > 1:
        #efficientnetb1 = torch.nn.parallel.DataParallel(efficientnetb1)
    #effientnetb1 = efficientnetb1.to('cuda')
    model_name = 'efficientnetb1_quant_distilled_{}bits'.format(numBit)
    model_path = os.path.join(imageNetmodelsFolder, model_name)

    if not model_name in imagenet_manager.saved_models:
        imagenet_manager.add_new_model(model_name, model_path)

    imagenet_manager.train_model(efficientnetb1, model_name=model_name,
                                 train_function=convForwModel.train_model,
                                 arguments_train_function={'epochs_to_train': epochsToTrainImageNet,
                                                           'learning_rate_style': 'imagenet',
                                                           'initial_learning_rate': 0.01,
                                                           'use_nesterov':True,
                                                           'initial_momentum':0.9,
                                                           'weight_decayL2':0,
                                                           'start_epoch': 0,
                                                           'print_every':30,
                                                           # 'use_distillation_loss':True,
                                                           # 'teacher_model': efficientnetb2,
                                                           'quantizeWeights':True,
                                                           'numBits':numBit,
                                                           'bucket_size':256,
                                                           'quantize_first_and_last_layer': False},
                                 train_loader=train_loader, test_loader=test_loader)
