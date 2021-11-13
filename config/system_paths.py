import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ==================================================================
# SET THESE PATHS MANUALLY #########################################
# ==================================================================

# ==================================================================
# project dirs
# ==================================================================
tensorboard_root = '/scratch_net/biwidl217_second/arismu/Tensorboard/'
project_root = '/scratch_net/biwidl217_second/arismu/Master_Thesis_Codes/project_TransUNet/TransUNet/'
data_root = '/scratch_net/biwidl217_second/arismu/Data_MT/'

# ==================================================================
# dirs where the pre-processed data is stored
# ==================================================================
preproc_folder_hcp = os.path.join(data_root,'HCP/')
preproc_folder_abide = os.path.join(data_root,'ABIDE/')
preproc_folder_nci = os.path.join(data_root,'NCI/')
preproc_folder_pirad_erc = os.path.join(data_root,'USZ/')
preproc_folder_promise = os.path.join(data_root,'PROMISE/')
preproc_folder_acdc = os.path.join(data_root,'ACDC/')
preproc_folder_mnms = os.path.join(data_root,'MnMs/')
preproc_folder_wmh = os.path.join(data_root,'WMH/')
preproc_folder_scgm = os.path.join(data_root,'SCGM/')

# ==================================================================
# define dummy paths for the original data storage. directly using pre-processed data for now
# ==================================================================
orig_data_root_acdc = ''
orig_data_root_nci = '/itet-stor/arismu/bmicdatasets-originals/Originals/Challenge_Datasets/NCI_Prostate/'
orig_data_root_promise = '/itet-stor/arismu/bmicdatasets-originals/Originals/Challenge_Datasets/Prostate_PROMISE12/'
orig_data_root_pirad_erc = '/itet-stor/arismu/bmicdatasets-originals/Originals/USZ/Prostate/'
orig_data_root_abide = '/itet-stor/arismu/bmicdatasets-originals/Originals/ABIDE/'
orig_data_root_hcp = '/itet-stor/arismu/bmicdatasets-originals/Originals/HCP/'
orig_data_root_mnms = '/itet-stor/arismu/bmicdatasets-originals/Originals/Challenge_Datasets/MnMs/'
#orig_data_root_wmh = '/cluster/work/cvl/shared/bmicdatasets/original/Challenge_Datasets/WMH_MICCAI2017/'
orig_data_root_scgm = '/itet-stor/arismu/bmicdatasets-originals/Originals/Challenge_Datasets/SCGM/'