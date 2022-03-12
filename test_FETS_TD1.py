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
from utils import test_single_volume, test_single_volume_FETS
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import config.system_paths as sys_config
import utils_data
from networks.unet_class import UNET
import utils
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
from calibration_functions import find_bin_values, find_bin_values_FETS
from calibration_functions import find_area

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='/itet-stor/arismu/bmicdatasets-originals/Originals/Challenge_Datasets/FeTS/', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--test_dataset', type=str,
                    default='FETS_TEST_TD1', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--max_iterations', type=int,default=6800, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=400, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=240, help='input patch size of network input')
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

seed = 1234
model_type = 'UNET'
data_aug = 0.5


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

    imts = np.array(imts)
    gtts = np.array(gtts)

    #print(f"Unique1: {np.unique(gtts)}")
    gtts[np.where(gtts == 4)] = 3
    #print(f"Unique2: {np.unique(gtts)}")


    model.eval()
    metric_list_whole = 0.0
    metric_list_enhancing = 0.0
    metric_list_core = 0.0
    pred_list = []
    label_list = []

    img_list = []
    label_list = []

    imts = torch.from_numpy(imts)
    gtts = torch.from_numpy(gtts)


    lim1 = 0
    lim2 = 155 
    lim3 = 310
    lim4 = 465
    x = []


    for i in range(imts.shape[2]):
        if lim1 != 0 and (lim1 % 155) == 0:
            lim1 = lim4
            lim2 = lim1 + 155
            lim3 = lim1 + 310
            lim4 = lim1 + 465
        x.append(imts[:, :, lim1])
        x.append(imts[:, :, lim2])
        x.append(imts[:, :, lim3])
        x.append(imts[:, :, lim4])
        y = torch.stack(x, dim = -1)
        img_list.append(y)


        lim1 += 1
        lim2 += 1
        lim3 += 1
        lim4 += 1
        x = []
        
        if lim4 == (imts.shape[2]):
            break

    
    lim1 = 0
    lim2 = 155 
    lim3 = 310
    lim4 = 465
    x = []


    for i in range(gtts.shape[2]):
        if lim1 != 0 and (lim1 % 155) == 0:
            lim1 = lim4
            lim2 = lim1 + 155
            lim3 = lim1 + 310
            lim4 = lim1 + 465
        x.append(gtts[:, :, lim1])
        x.append(gtts[:, :, lim2])
        x.append(gtts[:, :, lim3])
        x.append(gtts[:, :, lim4])
        y = torch.stack(x, dim = -1)
        label_list.append(y)


        lim1 += 1
        lim2 += 1
        lim3 += 1
        lim4 += 1
        x = []
        
        if lim4 == (gtts.shape[2]):
            break

        sub_subject_id_start_slice = 0
        subject_id_end_slice = 155

        first_bin = []
        second_bin = []
        third_bin = []
        fourth_bin = []
        fifth_bin = []

    
    for sub_num in range(num_test_subjects):


        # ============================
        # Group slices belonging to the same patients
        # ============================ 

        image = img_list[sub_subject_id_start_slice:subject_id_end_slice]
        label = label_list[sub_subject_id_start_slice:subject_id_end_slice]
        sub_subject_id_start_slice = subject_id_end_slice
        subject_id_end_slice = sub_subject_id_start_slice + 155


        image = torch.stack(image)
        label = torch.stack(label)

        """ subject_id_start_slice = np.sum(orig_data_siz_z[:sub_num])   #194 at the end of the loop
        subject_id_end_slice = np.sum(orig_data_siz_z[:sub_num+1])   #174 at the end of the loop
        image = imts[:,:, subject_id_start_slice:subject_id_end_slice] 
        label = gtts[:,:, subject_id_start_slice:subject_id_end_slice] 
 """
        #utils.save_nii(img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/' + 'FETS_TEST_TD1_test.nii.gz', data = image, affine = np.eye(4))

       
        image, label = image.cuda(), label.cuda()      
       

       
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

        metric_whole, metric_enhancing, metric_core, pred_l, label_l = test_single_volume_FETS(image, label, model, classes=args.num_classes, dataset = 'FETS_TD1', optim = 'ADAM', model_type = f'{model_type}', seed = '{seed}', patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=sub_num, z_spacing=args.z_spacing)

        logging.info("WHOLE TUMOR:")
        metric_list_whole += np.array(metric_whole)
        logging.info('case %s mean_dice %f mean_hd95 %f' % (sub_num, np.mean(metric_whole, axis=0)[0], np.mean(metric_whole, axis=0)[1]))

        logging.info("ENHANCING TUMOR:")
        metric_list_enhancing += np.array(metric_whole)
        logging.info('case %s mean_dice %f mean_hd95 %f' % (sub_num, np.mean(metric_enhancing, axis=0)[0], np.mean(metric_enhancing, axis=0)[1]))

        logging.info("TUMOR CORE:")
        metric_list_core += np.array(metric_core)
        logging.info('case %s mean_dice %f mean_hd95 %f' % (sub_num, np.mean(metric_core, axis=0)[0], np.mean(metric_core, axis=0)[1]))

        if sub_num < 10:
            first_bin_frac_pos, second_bin_frac_pos, third_bin_frac_pos, fourth_bin_frac_pos, fifth_bin_frac_pos = find_bin_values_FETS(pred_l, label_l)
            first_bin.append(first_bin_frac_pos)
            second_bin.append(second_bin_frac_pos)
            third_bin.append(third_bin_frac_pos)
            fourth_bin.append(fourth_bin_frac_pos)
            fifth_bin.append(fifth_bin_frac_pos)


    metric_list_whole = metric_list_whole / num_test_subjects   #get mean metrics for every class      
    metric_list_enhancing = metric_list_enhancing / num_test_subjects   #get mean metrics for every class
    metric_list_core = metric_list_core / num_test_subjects

    avg_first_bin_frac_pos = float(sum(first_bin)/len(first_bin))
    avg_second_bin_frac_pos = float(sum(second_bin)/len(second_bin))
    avg_third_bin_frac_pos = float(sum(third_bin)/len(third_bin))
    avg_fourth_bin_frac_pos = float(sum(fourth_bin)/len(fourth_bin))
    avg_fifth_bin_frac_pos = float(sum(fifth_bin)/len(fifth_bin))

    print("Fraction of positives:")
    print(f"First bin: {avg_first_bin_frac_pos}")
    print(f"Second bin: {avg_second_bin_frac_pos}")
    print(f"Third bin: {avg_third_bin_frac_pos}")
    print(f"Fourth bin: {avg_fourth_bin_frac_pos}")
    print(f"Fifth bin: {avg_fifth_bin_frac_pos}")

    find_area(avg_first_bin_frac_pos, avg_second_bin_frac_pos, avg_third_bin_frac_pos, avg_fourth_bin_frac_pos, avg_fifth_bin_frac_pos)


    y = [avg_first_bin_frac_pos, avg_second_bin_frac_pos, avg_third_bin_frac_pos, avg_fourth_bin_frac_pos, avg_fifth_bin_frac_pos]
    x = [0, 0.3, 0.5, 0.7, 1]
        

    fig, ax = plt.subplots()
    ref_line_label = "Perfectly calibrated"
    ax.plot([0, 1], [0, 1], "k:", label=ref_line_label)
    
    ax.set(xlabel="Mean predicted probability", ylabel="Fraction of positives")

    ax.plot(x, y, "s-", label = 'Classifier')[0]
    ax.legend(loc="lower right")
    

    fig.savefig(f'/scratch_net/biwidl217_second/arismu/Data_MT/plots/FETS_TD1_calibration_{model_type}_da{data_aug}.png')



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
        'FETS_TEST_TD1': {
            'volume_path': '/itet-stor/arismu/bmicdatasets-originals/Originals/Challenge_Datasets/FeTS/',
            'num_classes': 4,
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

    args.exp = 'TU_' + dataset_name + str(args.img_size) 
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
    net = UNET(in_channels = 4, out_channels = 4, features = [32, 64, 128, 256]).cuda()

    snapshot = os.path.join(f'/scratch_net/biwidl217_second/arismu/Master_Thesis_Codes/project_TransUNet/model/2022/FETS/{model_type}/', f'FETS_UNET_best_val_loss_seed1234_da0.5.pth')
    #if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'no_data_aug_' + 'epoch_' + str(args.max_epochs-1))

    # ============================
    # Load the trained parameters into the model
    # ============================  

    net.load_state_dict(torch.load(snapshot))

    #size = sum(p.numel() for p in net.parameters())
    #print(f'Number of parameters: {size}')

    # ============================
    # Logging
    # ============================ 

    snapshot_name = snapshot_path.split('/')[-1]
    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    # ============================
    # Save the predictions as nii files
    # ============================ 

    if args.is_savenii:
        args.test_save_dir = f'../predictions_2022/FETS/{model_type}/'
        test_save_path = os.path.join(args.test_save_dir, f'TD1_FETS_{model_type}_test_seed{seed}_da{data_aug}')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    inference(args, net, test_save_path)

print("test.py successfully executed")

