# PatternTrain: Fast and Accurate Deep CNN Training via Dynamic Fine-Grained Pattern-Based Pruning
---

## About PatternTrain
This repository contains the source codes and scripts for training a set of DNNs using PatternTrain, which is deisgned for (1) high training peformance (low training FLOPs), (2) high accuracy, and (3) high compression ratio (low inference FLOPs) by using dynamic fine-grained pattern-based pruning. We provide the codes to efficiently and accurately train ResNet and VGG models for CIFAR-10, CIFAR-100, and ImageNet (ILSVRC 2012) datasets.

## Prerequisites
```
Python 3.6+
PyTorch 1.0+
NVIDIA CUDA 9.0+
```

## Installation (via Anaconda)
- Download and install Anaconda
```
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
```

- Install dependencies
```
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
```

## Data Preparation
- Prepare CIFAR-10/100 datasets 
```
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzvf cifar-100-python.tar.gz
```

- Download ImageNet dataset at http://www.image-net.org/

## Parameter and Argument
- --data-path: path to dataset
- --dataset: supported dataset name (cifar10, cifar100, imagenet)
- --arch: supported neural network architecture (vgg11, vgg13, resnet32, resnet50)
- --num-gpus: the number of gpu(s)
- --gpu-list: the desired gpu(s) list

## Training Example
- Training ResNet-32 model using PatternTrain on CIFAR-10 dataset with two GPUs. 
```
python3 run_script.py --data-path /path/to/dataset --dataset cifar10 --arch resnet32 --num-gpus 2 --gpu-list 0 1
```

- Training ResNet-50 model using PatternTrain on ImageNet dataset with four GPUs.
```
python3 run_script.py --data-path /path/to/ilsvrc12 --dataset imagenet --arch resnet50 --num-gpus 4 --gpu-list 0 1 2 3
```
## Online Publiic Repository
The code and instructions can be also accessed at https://anonymous.4open.science/r/a756802d-6618-43ff-a170-51174b0ad7eb/.
