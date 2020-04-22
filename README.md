# PatternTrain: Fast and Accurate DNN Training via Dynamic Fine-Grained Pattern-Based Pruning
---
This repository is the software artifact for PatternTrain paper, which aims to use fine-grained pattern-based pruning to achieve (1) high training peformance (low training FLOPs), (2) high accuracy, and (3) high compression ratio (low inference FLOPs). The repository contains the codes to efficiently and accurately train ResNet and VGG models for CIFAR-10, CIFAR-100, and ImageNet (ILSVRC 2012) datasets.

## Prerequisites
```
Python 3.6+
PyTorch 1.0+
NVIDIA CUDA 9.0+
```

## Installation (via Anaconda)
```
bash Anaconda-latest-Linux-x86_64.sh
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch

```

## Parameter
```
--data-path: path to dataset
--dataset: supported dataset name (cifar10, cifar100, imagenet)
--arch: supported neural network architecture (vgg11, vgg13, resnet32 resnet50)
--num-gpus: the number of gpu(s)
--gpu-list: the desired gpu(s) list
```

## Training Example
- Training ResNet-32 model on CIFAR-10 dataset with two GPUs. 
```
python3 run_script.py --data-path /path/to/dataset --dataset cifar10 --arch resnet32 --num-gpus 2 --gpu-list 0 1
```

- Training ResNet-50 model on ImageNet dataset with four GPUs.
```
python3 run_script.py --data-path /path/to/ilsvrc12 --dataset imagenet --arch resnet50 --num-gpus 4 --gpu-list 0 1 2 3
```