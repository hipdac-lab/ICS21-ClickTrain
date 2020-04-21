import argparse
import os
import sys
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import yaml
import math
import random

from tricks import CrossEntropyLossMaybeSmooth, mixup_data, mixup_criterion
from utils import *

# import resnet32_flat_ori as resnet
import resnet
import vgg

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                    and name.startswith("resnet")
                    and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
#                     choices=model_names,
#                     help='model architecture: ' + ' | '.join(model_names) +
#                     ' (default: resnet18)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--multi-gpu', action='store_true', default=False,
                    help='multi-gpu training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batch size (default: 128)')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', nargs='+', default=[150, 225], type=int,
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit)')
parser.add_argument('--save-dir', dest='save_dir', default='save_temp', type=str,
                    help='The directory used to save the trained models')
parser.add_argument('--save-every', dest='save_every', default=10, type=int,
                    help='Saves checkpoints at every specified number of epochs')
parser.add_argument('--ite-enable-mixup', default=0, type=int,
                    metavar='N', help='iteration enable mixup (default: 0)')
parser.add_argument('--mixup', action='store_true', default=False,
                    help='cross entropy mixup')
parser.add_argument('--alpha', default=0.0, type=float, metavar='M',
                    help='mixup training, lambda = beta(alpha, alpha) distribution')
parser.add_argument('--smooth', action='store_true', default=False,
                    help='lable smooth')
parser.add_argument('--smooth-eps', default=0.0, type=float, metavar='M',
                    help='smoothing rate [0.0, 1.0]')
parser.add_argument('--num-patterns', default=4, type=int,
                    help='number of patterns (default is 4)')
parser.add_argument('--epoch-choose-p-k', default=2, type=int,
                    help='epoch of choosing patterns and kernels (default is 61)')
parser.add_argument('--epoch-hard-prune-p-k', default=4, type=int,
                    help='epoch of hard pruning patterns and kernels and kernels (default is 131)')
parser.add_argument('--grp-lasso-coeff', default=0.00025, type=float,
                    help='group lasso coefficient (default is 0.00025)')
parser.add_argument('--config-file', default='./configs/cifar10_resnet32_v1.yaml', type=str,
                    help='configuration file')


# manualSeed = ''
# Random seed
# if manualSeed is None:
# manualSeed = random.randint(1, 10000)
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)
# if True:
#     torch.cuda.manual_seed_all(manualSeed)

best_prec1 = 0
args = None


def main():
    global args, best_prec1
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    # Data
    print('==> Preparing dataset %s' % args.dataset)

    if args.dataset == 'cifar10':
        specific_datasets = datasets.CIFAR10
        data_path = './datasets/{}'.format(args.dataset)
        num_classes = 10
    else:
        specific_datasets = datasets.CIFAR100
        data_path = './datasets/{}'.format(args.dataset)
        num_classes = 100

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_loader = torch.utils.data.DataLoader(
        specific_datasets(root=data_path, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        specific_datasets(root=data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if 'vgg' in args.arch:
        models = vgg
    elif 'resnet' in args.arch:
        models = resnet

    model = models.__dict__[args.arch](num_classes=num_classes)

    if args.multi_gpu:
        model = torch.nn.DataParallel(model)

    model.cuda()

    for name, weight in model.named_parameters():
        # if 'conv' in name:
        print(name)
        print(weight.shape)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

        for name, weight in model.named_parameters():
            layer = 'name: {}; shape: {}'.format(name, weight.shape)
            print(layer)

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    args.smooth = False
    criterion = CrossEntropyLossMaybeSmooth(smooth_eps=args.smooth_eps).cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    milestones = []
    for step in args.schedule:
        milestones.append(step)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=milestones, 
                                                        last_epoch=args.start_epoch - 1)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    writer = SummaryWriter(log_dir='./logs_group', comment='parameters')

    p_masks_dic = {}
    k_masks_dic = {}
    p_masks_r_dic = {}
    k_masks_r_dic = {}

    t_acc_f = './t_acc_f.txt'
    if os.path.exists(t_acc_f):
        os.remove(t_acc_f)

    compression_ratio = 0
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch, p_masks_dic, k_masks_dic, p_masks_r_dic,
              k_masks_r_dic, writer)

        # adjust_learning_rate
        lr_scheduler.step()

        # evaluate on validation set
        loss, prec1 = test(model, criterion, val_loader)
        # prec1 = validate(val_loader, model, criterion)

        print('test irregular sparsity after evaluating on validation dataset')
        compression_ratio = test_irregular_sparsity(model)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        print('best_prec1: {}'.format(best_prec1))

        # from contextlib import redirect_stdout
        # with open(t_acc_f, "a") as acc_f:
        #     with redirect_stdout(acc_f):
        #         t_acc = '{} {} {}'.format(epoch, loss, prec1)
        #         print(t_acc)

        # if epoch > 0 and epoch % args.save_every == 0:
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'best_prec1': best_prec1,
        #     }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        if epoch == args.epochs - 1:
            print('Saving model: ./{}/{}_{}_acc_{}_ratio_{}.pt\n'.format(
                    args.save_dir, args.dataset, args.arch, best_prec1, compression_ratio))

            torch.save(model.state_dict(), "./{}/{}_{}_acc_{}_ratio_{}.pt".format(
                args.save_dir, args.dataset, args.arch, best_prec1, compression_ratio))


def gen_pattern_dic(num_patterns):

    # pattern1 = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0]]  # 3
    # pattern2 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]  # 12
    # pattern3 = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]]  # 65
    # pattern4 = [[0, 2], [1, 2], [2, 0], [2, 1], [2, 2]]  # 120

    # pattern5 = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]  # 1
    # pattern6 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 2]]  # 14
    # pattern7 = [[0, 0], [0, 2], [1, 0], [2, 0], [2, 2]]  # 44
    # pattern8 = [[0, 0], [0, 2], [1, 2], [2, 0], [2, 2]]  # 53
    #
    # pattern9 = [[1, 0], [1, 2], [2, 0], [2, 1], [2, 2]]  # 125
    # pattern10 = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2]]  # 6
    # pattern11 = [[1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]  # 126
    # pattern12 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 0]]  # 10

    pattern1 = [[0., 0., 0.], [0., 1., 1.], [0., 1., 1.]]
    pattern2 = [[0., 0., 0.], [1., 1., 0.], [1., 1., 0.]]
    pattern3 = [[0., 1., 1.], [0., 1., 1.], [0., 0., 0.]]
    pattern4 = [[1., 1., 0.], [1., 1., 0.], [0., 0., 0.]]

    pattern5 = [[0., 0., 0.], [0., 0., 1.], [1., 1., 1.]]
    pattern6 = [[0., 0., 0.], [1., 1., 1.], [0., 1., 0.]]
    pattern7 = [[0., 1., 0.], [0., 1., 1.], [0., 1., 0.]]
    pattern8 = [[0., 1., 0.], [1., 1., 0.], [0., 1., 0.]]

    pattern9 = [[1., 1., 1.], [0., 1., 0.], [0., 0., 0.]]
    pattern10 = [[0., 0., 0.], [1., 0., 0.], [1., 1., 1.]]
    pattern11 = [[1., 1., 1.], [1., 0., 0.], [0., 0., 0.]]
    pattern12 = [[0., 0., 0.], [1., 0., 1.], [0., 1., 1.]]

    patterns_dict = {}
    if num_patterns == 4:
        patterns_dict = {0: pattern1,
                         1: pattern2,
                         2: pattern3,
                         3: pattern4
                         }
    elif num_patterns == 8:
        patterns_dict = {0: pattern1,
                         1: pattern2,
                         2: pattern3,
                         3: pattern4,
                         4: pattern5,
                         5: pattern6,
                         6: pattern7,
                         7: pattern8
                         }
    elif num_patterns == 12:
        patterns_dict = {0: pattern1,
                         1: pattern2,
                         2: pattern3,
                         3: pattern4,
                         4: pattern5,
                         5: pattern6,
                         6: pattern7,
                         7: pattern8,
                         8: pattern9,
                         9: pattern10,
                         10: pattern11,
                         11: pattern12
                         }

    return patterns_dict


def train(train_loader, model, criterion, optimizer, epoch, p_masks_dic, k_masks_dic, p_masks_r_dic,
          k_masks_r_dic, writer):
    '''
    Run one train epoch
    '''

    config = args.config_file

    if not isinstance(config, str):
        raise Exception("filename must be a string")
    with open(config, "r") as stream:
        try:
            raw_dict = yaml.safe_load(stream)
            prune_ratios = raw_dict['prune_ratios']
        except yaml.YAMLError as exc:
            print(exc)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for idx, (inputs, targets) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input_var = inputs.cuda()
        target_var = targets.cuda()

        if args.half:
            input_var = input_var.half()

        # determine the condition to update mask
        # p means pattern, k means kernel, c means channel, f means filter

        epoch_update_p_k = args.epoch_choose_p_k
        epoch_hard_prune_p_k = args.epoch_hard_prune_p_k

        if epoch < epoch_update_p_k:
            update_mode = 0
        elif epoch == epoch_update_p_k and idx == 0:
            update_mode = 1
        elif epoch_update_p_k <= epoch < epoch_hard_prune_p_k:
            update_mode = 2
        elif epoch >= epoch_hard_prune_p_k:
            update_mode = 3
            
            if idx >= args.ite_enable_mixup:
                args.mixup = True
        else:
            update_mode = 0
            args.mixup = False

        patterns_dict = gen_pattern_dic(args.num_patterns)

        weight_l2_cat = torch.tensor([]).cuda()
        p_weight_l2_cat = torch.tensor([]).cuda()
        k_weight_l2_cat = torch.tensor([]).cuda()

        # group lasso regularization
        group_lasso_reg_p = torch.tensor(0.).cuda()
        group_lasso_reg_k = torch.tensor(0.).cuda()

        if update_mode == 0:
            group_lasso_reg_p = 0
            group_lasso_reg_k = 0

        elif update_mode == 1:
            # compute output for gradient
            output = model(input_var)
            loss = criterion(output, target_var)
            optimizer.zero_grad()
            loss.backward()

            print("update k mask based on weight and gradient")
            for (name, weight) in model.named_parameters():
                # ignore layers that do not have rho
                if not args.multi_gpu:
                    name = 'module.{}'.format(name)

                if name not in prune_ratios:
                    continue

                # the percent from script is used to represent k-th smallest importance kernel
                # will be pruned in each filter
                shape = weight.shape
                kth_smallest = math.ceil(prune_ratios[name] * shape[0] * shape[1] / 100)

                mask = torch.ones(shape, dtype=torch.float32)
                mask_r = torch.zeros(shape, dtype=torch.float32)

                with torch.no_grad():
                    # for i in range(shape[0]):
                    #     kernel_imp_list = (weight[i, :, :, :] * weight.grad[i, :, :, :]).pow(2).sum([1, 2]).reshape((-1,))
                    #     _, kth_smallest_index = torch.topk(kernel_imp_list, k=kth_smallest, largest=False)
                    
                    #     for j in kth_smallest_index:
                    #         mask[i, j, :, :] = torch.zeros((shape[2], shape[3]), dtype=torch.float32)
                    #         mask_r[i, j, :, :] = torch.ones((shape[2], shape[3]), dtype=torch.float32)

                    kernel_imp_list = (weight * weight.grad).pow(2).sum([2, 3]).reshape((-1,))
                    _, kth_smallest_index = torch.topk(kernel_imp_list, k=kth_smallest, largest=False)

                    for k_idx in kth_smallest_index:
                        j = k_idx % shape[1]
                        i = (k_idx - j) / shape[1]
                        mask[i, j, :, :] = torch.zeros((shape[2], shape[3]), dtype=torch.float32)
                        mask_r[i, j, :, :] = torch.ones((shape[2], shape[3]), dtype=torch.float32)

                    k_masks_dic[name] = mask
                    k_masks_r_dic[name] = mask_r

                mask_r_gpu = mask_r.cuda()
                weight_l2 = (weight * mask_r_gpu).pow(2).sum([2, 3]).reshape((-1,))
                weight_l2_cat = torch.cat((weight_l2_cat, weight_l2))
            group_lasso_reg_k = weight_l2_cat.add(1.0e-8).sqrt().sum()

            print("update p mask based on weight and gradient")
            weight_l2_cat = torch.tensor([]).cuda()
            for (name, weight) in model.named_parameters():
                shape = weight.shape
                # ignore layers that do not have rho
                if not args.multi_gpu:
                    name = 'module.{}'.format(name)

                if name not in prune_ratios:
                    continue
                elif shape[2] < 3:
                    continue

                pattern_imp = torch.zeros((args.num_patterns, shape[0], shape[1]), dtype=torch.float32)
                mask = np.ones(shape, dtype=np.float32)
                with torch.no_grad():
                    for key, pattern in patterns_dict.items():
                        pattern = torch.tensor(pattern).cuda()
                        pattern_imp[key, :, :] = (weight * weight.grad * pattern).pow(2).sum([2, 3])

                    best_pattern_idx = torch.argmax(pattern_imp, dim=0)

                    # print(best_pattern_idx)
                    # count_pattern_distribution(best_pattern_idx.view(-1).numpy())

                    shape_b = best_pattern_idx.shape
                    for i in range(shape_b[0]):
                        for j in range(shape_b[1]):
                            dict_idx = best_pattern_idx[i, j]
                            mask[i, j, :, :] = patterns_dict[dict_idx.item()]

                    mask_tmp = np.subtract(mask, 1.0)
                    mask_r = np.where(mask_tmp < 0, 1.0, mask_tmp).astype(np.float32)

                p_masks_dic[name] = mask
                p_masks_r_dic[name] = mask_r
                mask_r_gpu = torch.from_numpy(mask_r).cuda()
                weight_l2 = (weight * mask_r_gpu).pow(2).sum([2, 3]).reshape((-1,))
                weight_l2_cat = torch.cat((weight_l2_cat, weight_l2))
            group_lasso_reg_p = weight_l2_cat.add(1.0e-8).sqrt().sum()

            # print("place patterns on remaining kernels")
            # for (name, weight) in model.named_parameters():
            #     # ignore layers that do not have rho
            #     if not args.multi_gpu:
            #         name = 'module.{}'.format(name)
            #
            #     if name not in prune_ratios:
            #         continue
            #
            #     p_masks_dic[name] *= k_masks_dic[name].numpy()
            #     p_masks_r_dic[name] *= k_masks_dic[name].numpy()

        elif update_mode == 2:
            # print("regularization using masks_dic")
            p_weight_l2_cat = torch.tensor([]).cuda()
            k_weight_l2_cat = torch.tensor([]).cuda()

            for (name, weight) in model.named_parameters():
                # ignore layers that do not have rho
                if not args.multi_gpu:
                    name = 'module.{}'.format(name)

                if name not in prune_ratios:
                    continue

                shape = weight.shape
                if shape[2] < 3:
                    p_masks_r_gpu = torch.zeros(shape, dtype=torch.float32).cuda()
                else:
                    p_masks_r_gpu = torch.from_numpy(p_masks_r_dic[name]).cuda()

                weight_l2 = (weight * p_masks_r_gpu).pow(2).sum([2, 3]).reshape((-1,))
                p_weight_l2_cat = torch.cat((p_weight_l2_cat, weight_l2))

                k_masks_r_gpu = k_masks_r_dic[name].cuda()
                weight_l2 = (weight * k_masks_r_gpu).pow(2).sum([2, 3]).reshape((-1,))
                k_weight_l2_cat = torch.cat((k_weight_l2_cat, weight_l2))

            group_lasso_reg_p = p_weight_l2_cat.add(1.0e-8).sqrt().sum()
            group_lasso_reg_k = k_weight_l2_cat.add(1.0e-8).sqrt().sum()

        if not update_mode == 1:
            if args.mixup:
            # input = input.cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)
                args.smooth = False
                input_var, target_a, target_b, lam = mixup_data(input_var, target_var, args.alpha)

            # compute output
            output = model(input_var)

            if args.mixup:
                loss = mixup_criterion(criterion, output, target_a, target_b, lam, args.smooth)
            else:
                # loss = criterion(output, target, smooth=args.smooth)
                loss = criterion(output, target_var)

        if update_mode == 2:
            # if idx == 0:
            #     print('grp_lasso_coeff: {}'.format(args.grp_lasso_coeff))
            grp_lasso_coeff = args.grp_lasso_coeff
            total_loss = loss + grp_lasso_coeff * (group_lasso_reg_p + group_lasso_reg_k)
        else:
            total_loss = loss

        # compute gradient and do SGD step
        if not update_mode == 1:
            optimizer.zero_grad()
            total_loss.backward()

        optimizer.step()

        if update_mode == 3:
            # print("update_mode == 3")
            for name, weight in (model.named_parameters()):
                shape = weight.shape

                if not args.multi_gpu:
                    name = 'module.{}'.format(name)

                if name not in prune_ratios:
                    continue
                elif shape[2] < 3:
                    continue

                mask_gpu = torch.from_numpy(p_masks_dic[name]).cuda()
                with torch.no_grad():
                    weight.data = weight * mask_gpu

            for name, weight in (model.named_parameters()):
                if not args.multi_gpu:
                    name = 'module.{}'.format(name)

                if name not in prune_ratios:
                    continue

                mask_gpu = k_masks_dic[name].cuda()
                with torch.no_grad():
                    weight.data = weight * mask_gpu

            if idx == 0:
                test_irregular_sparsity(model)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, idx, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

        # w_epoch = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 150]
        # if epoch in w_epoch and idx == 0:
        #     w_file = './{}_w_file.txt'.format(epoch)
        #     if os.path.exists(w_file):
        #         os.remove(w_file)

        #     from contextlib import redirect_stdout
        #     with open(w_file, "a") as w_f:
        #         with redirect_stdout(w_f):
        #             for name, weight in model.named_parameters():
        #                 name_l = 'name: {}'.format(name)
        #                 print(name_l)
        #                 print(weight)

        # if idx % 100 == 0:
        #     idx_writer = epoch * len(train_loader) + idx
        #     writer.add_scalar('Accuracy/training_acc', prec1, idx_writer)
        #     writer.add_scalar('Loss/loss', loss, idx_writer)
        #     writer.add_scalar('Loss/total_loss', total_loss, idx_writer)
        #
        #     for (name, weight) in model.named_parameters():
        #         # ignore layers that do not have rho
                # if not args.multi_gpu:
                #     name = 'module.{}'.format(name)

        #         if name not in prune_ratios:
                # if args.multi_gpu:
                #     name = 'module.{}'.format(name)
        #             continue
        #
        #         writer.add_histogram(name, weight, idx_writer)
        #     writer.close()


def test(model, criterion, test_loader):
    model.eval()
    losses = AverageMeter()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            losses.update(loss.item(), data.size(0))
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            output_max, output_max_i = output.max(1, keepdim=True)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nTest set loss: {:.4f},  * Acc@1: {}/{} ({:.2f}%)\n'.format(
        losses.avg, correct, len(test_loader.dataset),
        100. * float(correct) / float(len(test_loader.dataset))))
    return losses.avg, (100. * float(correct) / float(len(test_loader.dataset)))


def validate(val_loader, model, criterion):
    '''
    Run evaluation
    '''

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()