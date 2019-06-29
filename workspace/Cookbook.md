# Misc cookbooks

*Started by the Officially Forgetful*

## Distiller

How to get all weights' names without getting your hands dirty:
`python3 compress_classifier.py -a=efficientnet_cifar_b0 ../../../data.cifar10 --summary=sparsity`
and `>> res = ['module.'+w for w in model.state_dict().keys() if (w.endswith(".weight") and "_bn" not in w)]; >> assert len(res) == [number of rows in the table above]`

## SFTP + GCloud

Install and configure [gcloud CLI tool](https://cloud.google.com/compute/docs/gcloud-compute/) on the local machine. Run `gcloud compute config-ssh` to create SSH configuration file in `~/.ssh/config` (or previously chosen location). Then use the hostname in the resulting config to ssh (from the local machine) into the instance and find out the username (`username@hostname`). Add `username` to `~/.ssh/config` as `User username`. After this, you should be able to use `ssh` as normal by `ssh username@hostname` without being prompted for the password and "Permission denied (publickey).". Enter these hostname and username into `sftp-config.json`, and you should be done.

## Setup from scratch (maybe transfer this to the README)

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda 
rm ~/miniconda.sh
echo "PATH=$PATH:$HOME/miniconda/bin" >> .bashrc
exec bash
conda create -n micronet python=3.7 anaconda  # and respond yes here
conda activate micronet
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch  # pytorch with CUDA=10
```

now move on with the instructions from the README.
