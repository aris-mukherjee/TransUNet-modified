import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.unet_class import UNET
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_runmc

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/itet-stor/arismu/bmicdatasets-originals/Originals/Challenge_Datasets/NCI_Prostate/', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='RUNMC', help='experiment_name')
#parser.add_argument('--list_dir', type=str,
#                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=6800, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=400, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=1e-3,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--target_resolution', type=float, default=0.625, help='target resolution')                    


parser.add_argument('--tr_run_number', type = int, default = 3) # 1 / 
parser.add_argument('--tr_cv_fold_num', type = int, default = 1) # 1 / 2
parser.add_argument('--da_ratio', type = float, default = 0.0) # 0.0 / 0.25

args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'RUNMC': {
            'root_path': '/itet-stor/arismu/bmicdatasets-originals/Originals/Challenge_Datasets/NCI_Prostate/',
            'num_classes': 3,
            'target_resolution': 0.625
        },
    }


    if args.batch_size != 16 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 16
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.target_resolution = dataset_config[dataset_name]['target_resolution']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)

    # ===========================    
    # define snapshot path where model will be stored
    # ===========================             

    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 6800 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 400 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 1e-3 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    # ===========================    
    # create an instance of the model 
    # ===========================      
    
    #net = UNet_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)#.cuda()
    #net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)#.cuda()
    #net.load_from(weights=np.load(config_vit.pretrained_path))

    net = UNET(in_channels = 3, out_channels = 3, features = [64, 128, 256, 512]).cuda()

    # ===========================    
    # start training 
    # ===========================  

    trainer = {'RUNMC': trainer_runmc,}
    trainer[dataset_name](args, net, snapshot_path)

