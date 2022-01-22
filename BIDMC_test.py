import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import config.system_paths as sys_config
import utils_data
from networks.unet_class import UNET
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='/itet-stor/arismu/bmicdatasets-originals/Originals/Challenge_Datasets/Prostate_PROMISE12/TrainingData/', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--test_dataset', type=str,
                    default='BIDMC', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--max_iterations', type=int,default=6800, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=400, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
#parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--is_savenii', type=bool, default=True, help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=1e-3, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--target_resolution', type=float, default=0.625, help='target resolution')   
parser.add_argument('--image_depth_tr', type=int, default=32, help='target resolution') 
parser.add_argument('--image_depth_ts', type=int, default=32, help='target resolution')    
parser.add_argument('--test_cv_fold_num', type = int, default = 1) # 1 / 2 / 3 / 4
parser.add_argument('--NORMALIZE', type = int, default = 1) # 1 / 0
args = parser.parse_args()



def inference(args, model, test_save_path=None):

    # ============================
    # Load test data
    # ============================ 
    
    loaded_test_data = utils_data.load_testing_data(args.test_dataset,  #needs to be adapted for different test set
                                                    args.test_cv_fold_num,
                                                    args.img_size,
                                                    args.target_resolution,
                                                    args.image_depth_ts)


    imts = loaded_test_data[0]   #shape (194, 256, 256)
    gtts = loaded_test_data[1]
    orig_data_res_x = loaded_test_data[2]
    orig_data_res_y = loaded_test_data[3]
    orig_data_res_z = loaded_test_data[4]
    orig_data_siz_x = loaded_test_data[5]
    orig_data_siz_y = loaded_test_data[6]
    orig_data_siz_z = loaded_test_data[7]
    name_test_subjects = loaded_test_data[8]
    num_test_subjects = loaded_test_data[9]
    ids = loaded_test_data[10]

    

    model.eval()
    metric_list = 0.0

    
    for sub_num in range(num_test_subjects):


        # ============================
        # Group slices belonging to the same patients
        # ============================ 

        subject_id_start_slice = np.sum(orig_data_siz_z[:sub_num])   #194 at the end of the loop
        subject_id_end_slice = np.sum(orig_data_siz_z[:sub_num+1])   #174 at the end of the loop
        image = imts[:,:, subject_id_start_slice:subject_id_end_slice] 
        label = gtts[:,:, subject_id_start_slice:subject_id_end_slice] 

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        image, label = image.cuda(), label.cuda()      
        image = image.permute(2, 0, 1)
        label = label.permute(2, 0, 1)

        image = torch.rot90(image, 1, [1, 2])
        label = torch.rot90(label, 1, [1, 2])

        # ==================================================================
        # setup logging
        # ==================================================================
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        subject_name = str(name_test_subjects[sub_num])[2:-1]
        logging.info('============================================================')
        logging.info('Subject ' + str(sub_num+1) + ' out of ' + str(num_test_subjects) + ': ' + subject_name)

        # ============================
        # Perform the prediction for each test patient individually & calculate dice score and Hausdorff distance
        # ============================ 

        metric_i = test_single_volume(image, label, model, classes=args.num_classes, dataset = 'BIDMC', optim = 'ADAM', model_type = 'UNET_DROPOUT', seed= '100', patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=sub_num, z_spacing=args.z_spacing)

        metric_list += np.array(metric_i)
        logging.info('case %s mean_dice %f mean_hd95 %f' % (sub_num, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / num_test_subjects   #get mean metrics for every class

        # ============================
        # Log the mean performance achieved for each class
        # ============================ 

    for i in range(0, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i][0], metric_list[i][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


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

    dataset_config = {
        'BIDMC': {
            'volume_path': '/itet-stor/arismu/bmicdatasets-originals/Originals/Challenge_Datasets/Prostate_PROMISE12/TrainingData/',
            'num_classes': 3,
            'z_spacing': 1,
        },
    }
    dataset_name = args.test_dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_name
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # ============================
    # Same snapshot path as defined in the train script to access the trained model
    # ============================  

    args.exp = 'TU_RUNMC' + str(args.img_size) 
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 6800 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 400 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 1e-3 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    #net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net = UNET(in_channels = 3, out_channels = 3, features = [32, 64, 128, 256]).cuda()

    snapshot = os.path.join('/scratch_net/biwidl217_second/arismu/Master_Thesis_Codes/project_TransUNet/model/2022/UNET/', 'UNET_DROPOUT_best_val_loss_seed100.pth')
    #f not os.path.exists(snapshot): snapshot = snapshot.replace('best_model',  'epoch_' + str(args.max_epochs-1))

    # ============================
    # Load the trained parameters into the model
    # ============================  

    net.load_state_dict(torch.load(snapshot))

    # ============================
    # Logging
    # ============================ 

    snapshot_name = snapshot_path.split('/')[-1]
    log_folder = './test_log/test_log_' + 'TU_BIDMC256'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    # ============================
    # Save the predictions as nii files
    # ============================ 

    if args.is_savenii:
        args.test_save_dir = '../predictions_2022/UNET/'
        test_save_path = os.path.join(args.test_save_dir, 'BIDMC_UNET_DROPOUT_test_seed100')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)

print("test.py successfully executed")

