"""Generate names for writing into *.yaml"""

from distiller.models import create_model
import pprint

pretrained = False
dataset = "cifar10"
arch = "efficientnetb0_cifar"
value = 1e-5

model = create_model(pretrained, dataset, arch)

all_weights = ['module.'+w for w in model.state_dict().keys() if (w.endswith(".weight") and "_bn" not in w)]
# filter out depthwise convs which we don't regularize:
regularized_weights = [name for name in all_weights if "_depthwise_conv" not in name]
reg_schedule = [f"{name}: {(value if name in regularized_weights else 0.)}"
                for name in all_weights]

print("\tAll weights:")
# pprint.pprint(all_weights)

print("\tRegularization")
for entry in reg_schedule:
    print(entry)
