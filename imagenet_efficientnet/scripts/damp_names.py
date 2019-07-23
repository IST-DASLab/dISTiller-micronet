from distiller.models import create_model
import pprint

pretrained = False
dataset = "imagenet"
arch = "efficientnetb1"
value = 1e-5

restricted = ['_bn', '.bias','project', 'expand']

model = create_model(pretrained, dataset, arch)
with open('names.txt', 'w+') as f:
    for name, param in model.named_parameters():
        #if not any([val in name for val in restricted]):
        if "_se_expand" in name and '.bias' not in name:
            f.write(f'"{name}",\n')