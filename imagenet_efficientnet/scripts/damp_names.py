from distiller.models import create_model
import pprint

pretrained = False
dataset = "imagenet"
arch = "efficientnetb1"
value = 1e-5

model = create_model(pretrained, dataset, arch)
with open('names.txt', 'w+') as f:
    for name, param in model.named_parameters():
        f.write(f'{name}\n')