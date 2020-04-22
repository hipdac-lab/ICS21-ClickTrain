PatternTrain: Fast and Accurate DNN Training via Dynamic Fine-Grained Pattern-Based Pruning
---

# Prerequisites
```
Python 3.6+
PyTorch 1.0+
```

# Installation
```
bash Anaconda-latest-Linux-x86_64.sh
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch

```

# Parameter
```
--data-path: path to dataset
--dataset: supported dataset name (cifar10, cifar100, imagenet)
--arch: supported neural network architecture (vgg11, vgg13, resnet32 resnet50)
--num-gpus: the number of gpu(s)
--gpu-list: the desired gpu(s) list
```

# Run example
```
python run_script.py --data-path /path/to/dataset --dataset cifar10 --arch resnet32 --num-gpus 2 --gpu-list 0 1

python run_script.py --data-path /path/to/ilsvrc12 --dataset imagenet --arch resnet50 --num-gpus 4 --gpu-list 0 1 2 3
```
