import os
import numpy as np
import logging
import utils
import subprocess

training_folder = '/itet-stor/arismu/bmicdatasets-originals/Originals/Challenge_Datasets/FeTS/MICCAI_FeTS2021_TrainingData/'

def do_bias_correction(all_indices):


    for folder in os.listdir(training_folder):
        if not (folder.lower().endswith('.csv') or folder.lower().endswith('.md')):
            patient_id = int(folder.split('_')[-1])  
            patname = folder
            nifti_img_path = '/scratch_net/biwidl217_second/arismu/Data_MT/FeTS' + '/Individual_NIFTI/' + patname + '/'

            if not os.path.exists(nifti_img_path):
                if patient_id in all_indices:
                    image_t1, _, image_t1_hdr = utils.load_nii(training_folder + folder + f'/{patname}_t1.nii.gz')
                    image_t1ce, _, image_t1ce_hdr = utils.load_nii(training_folder + folder + f'/{patname}_t1ce.nii.gz')
                    image_t2, _, image_t2_hdr = utils.load_nii(training_folder + folder + f'/{patname}_t2.nii.gz')
                    image_flair, _, image_flair_hdr = utils.load_nii(training_folder + folder + f'/{patname}_flair.nii.gz')

                    
                    if not os.path.exists(nifti_img_path):
                        utils.makefolder(nifti_img_path)
                    if os.path.isfile(nifti_img_path + patname + '_img_t1.nii.gz'):
                        pass
                    else:
                        utils.save_nii(img_path = nifti_img_path + patname + '_img_t1.nii.gz', data = image_t1, affine = np.eye(4))
                        utils.save_nii(img_path = nifti_img_path + patname + '_img_t1ce.nii.gz', data = image_t1ce, affine = np.eye(4))
                        utils.save_nii(img_path = nifti_img_path + patname + '_img_t2.nii.gz', data = image_t2, affine = np.eye(4))
                        utils.save_nii(img_path = nifti_img_path + patname + '_img_flair.nii.gz', data = image_flair, affine = np.eye(4))

                    input_img_t1 = nifti_img_path + patname + '_img_t1.nii.gz'
                    output_img_t1 = nifti_img_path + patname + '_img_t1_n4.nii.gz'
                    input_img_t1ce = nifti_img_path + patname + '_img_t1ce.nii.gz'
                    output_img_t1ce = nifti_img_path + patname + '_img_t1ce_n4.nii.gz'
                    input_img_t2 = nifti_img_path + patname + '_img_t2.nii.gz'
                    output_img_t2 = nifti_img_path + patname + '_img_t2_n4.nii.gz'
                    input_img_flair = nifti_img_path + patname + '_img_flair.nii.gz'
                    output_img_flair = nifti_img_path + patname + '_img_flair_n4.nii.gz'

                    for input_img, output_img in zip([input_img_t1, input_img_t1ce, input_img_t2, input_img_flair], [output_img_t1, output_img_t1ce, output_img_t2, output_img_flair]):
                        if os.path.isfile(output_img):
                            img = utils.load_nii(img_path = output_img)[0]
                        else:
                            subprocess.call(["/itet-stor/arismu/bmicdatasets_bmicnas01/Sharing/N4_th", input_img, output_img])
                            img = utils.load_nii(img_path = output_img)[0]
                    

    print("All subjects preprocessed!")

if __name__ == "__main__":

    logging.info('Counting files and parsing meta data...')

    training_ids = [1, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 83, 97, 82, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 81, 
                    98, 99, 100, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109,
                    108, 107, 106, 105, 104, 103, 102, 101, 67, 66, 84, 64, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15,
                    65, 13, 12, 11, 10, 9]

    validation_ids = [ 54, 53, 51, 50, 49, 52, 47, 35, 36, 37, 38, 48, 40, 41, 39, 43, 44, 45, 46, 42]

    test_ids_sd = [8, 7, 6, 5, 4, 3, 2, 31, 32, 14, 22, 63, 62, 61, 60, 59, 58, 57, 56, 55]

    test_ids_td1= [351, 352, 353, 354, 355, 356, 357, 358, 359, 361, 367, 362, 363, 364, 365, 366, 350, 360, 349, 369, 347, 368, 336, 337, 
                        348, 339, 340, 338, 342, 343, 344, 345, 346, 341]
 
    test_ids_td2 = [204, 199, 200, 201, 202, 203, 206, 211, 208, 209, 210, 198, 212, 213, 207, 197, 205, 195, 181, 182, 183, 184, 185, 187, 188, 
                186, 189, 190, 191, 192, 193, 194, 180, 196]

    all_indices =  training_ids + validation_ids + test_ids_sd + test_ids_td1 + test_ids_td2

    do_bias_correction(all_indices)

