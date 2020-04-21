Example:#python run_script.py --data-path /path/to/dataset --dataset cifar10 --arch resnet32 --num-gpus 2 --gpu-list 0 1

#python run_script.py --data-path /path/to/ilsvrc12 --dataset imagenet --arch resnet50 --num-gpus 4 --gpu-list 0 1 2 3

parameter: 
--data-path: path to dataset
--dataset: cifar10, cifar100, imagenet
--arch: vgg11, vgg13, resnet32 resnet50
--num-gpus: the number of gpu(s)
--gpu-list: the desire gpu(s) list