import os
import yaml
import argparse


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, required=True, help='path to validation dataset')
parser.add_argument('--dataset', default='imagenet', type=str, 
                    choices=['imagenet', 'cifar10', 'cifar100'], help='dataset name')
parser.add_argument('--arch', default = 'resnet50', type=str, help='arch name')
parser.add_argument('--config-version', default=1, type=int, help='version of config file')
parser.add_argument('--num-gpus', default=1, type=int, help='number of GPU(s) used in training')
parser.add_argument('--gpu-list', nargs='+', default=[0], type=int, help='list of GPU(s)')
args = parser.parse_args()

# Load configuration
if args.dataset == 'imagenet':
    assert args.arch in ['resnet50', 'vgg16']
    cfg_file = "imagenet_{}_v{}.yaml".format(args.arch, args.config_version)
    runfile = 'python imagenet.py'
elif args.dataset in ['cifar10', 'cifar100']:
    assert args.arch in ['resnet50', 'resnet32', 'vgg16', 'vgg13', 'vgg11']
    cfg_file = "{}_{}_v{}.yaml".format(args.dataset, args.arch, args.config_version)
    runfile = 'python cifar.py'
else:
    raise ValueError("{} is a wrong dataset".format(args.dataset))

with open(os.path.join('configs/', cfg_file)) as f_config:
    cfg = yaml.safe_load(f_config)

cfg['base']['learning-rate'] = cfg['base']['learning_rate'] * args.num_gpus
cfg['base']['train_batch'] = int(cfg['base']['train_batch'] * args.num_gpus)
if args.dataset == 'imagenet':
    cfg['base']['learning-rate'] /= 4
    cfg['base']['train_batch'] = int(cfg['base']['train_batch'] / 4)

# Data parallelism setup
gpu_id = ''
if args.num_gpus != len(args.gpu_list):
    print('gpu-list is not complete.')
else:
    i = 0
    for id in args.gpu_list:
        if i == 0:
            i += 1
            gpu_id += str(id)
        else:
            i += 1
            gpu_id += ',' + str(id)

cmd_line = runfile
cmd_line += ' --arch '                  +args.arch
cmd_line +=                             ' --multi-gpu' if len(args.gpu_list) >= 2 else ''
cmd_line += ' --gpu-id '                +gpu_id
cmd_line += ' --data-path '             +args.data_path if args.dataset == 'imagenet' else ''
cmd_line += ' --dataset '               +args.dataset if args.dataset.startswith('cifar') else ''
cmd_line += ' --workers '               +str(cfg['base']['workers'])
cmd_line += ' --epochs '                +str(cfg['base']['epochs'])
cmd_line += ' --learning-rate '         +str(cfg['base']['learning_rate'])
cmd_line += ' --schedule '              +str(cfg['base']['schedule'])
cmd_line += ' --train-batch '           +str(cfg['base']['train_batch'])
cmd_line += ' --test-batch '            +str(cfg['base']['test_batch'])
cmd_line += ' --print-freq '            +str(cfg['base']['print_freq'])
cmd_line += ' --save-dir '              +str(cfg['base']['save_dir'])
cmd_line += ' --save-every '            +str(cfg['base']['save_every'])
cmd_line += ' --ite-enable-mixup '      +str(cfg['base']['ite_enable_mixup'])
cmd_line += ' --alpha '                 +str(cfg['base']['alpha'])
cmd_line += ' --smooth-eps '            +str(cfg['base']['smooth_eps'])
cmd_line += ' --config-file '           +str(os.path.join('configs/', cfg_file))
cmd_line += ' --num-patterns '          +str(cfg['pt']['num_patterns'])
cmd_line += ' --epoch-choose-p-k '      +str(cfg['pt']['epoch_choose_p_k'])
cmd_line += ' --epoch-hard-prune-p-k '  +str(cfg['pt']['epoch_hard_prune_p_k'])
cmd_line += ' --grp-lasso-coeff '       +str(cfg['pt']['grp_lasso_coeff'])

print(cmd_line)
os.system(cmd_line)
