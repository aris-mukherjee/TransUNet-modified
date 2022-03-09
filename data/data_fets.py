import os
import numpy as np
import logging
import utils
import config.system_paths as sys_config
from skimage.transform import rescale
import gc
import h5py
import subprocess


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5

# ===============================================================
# This function unzips and pre-processes the data if this has not already been done.
# If this already been done, it reads the processed data and returns it.                    
# ===============================================================                         
def load_test_sd_data(input_folder,
              preproc_folder,
              size,
              target_resolution,
              force_overwrite = False):

    # ===============================
    # create the pre-processing folder, if it does not exist
    # ===============================
    utils.makefolder(preproc_folder)    
    
    # ===============================
    # file to create or directly read if it already exists
    # ===============================

    test_sd_file_name = 'data_test_sd_NEW.hdf5' 
    test_sd_file_path = os.path.join(preproc_folder, test_sd_file_name)
    
    # ===============================
    # if the images have not already been extracted, do so
    # ===============================
    if not os.path.exists(test_sd_file_path) or force_overwrite:
        
        logging.info('This configuration of protocol and data indices has not yet been preprocessed')
        logging.info('Preprocessing now...')
        prepare_test_sd_data(input_folder,
                     preproc_folder,
                     test_sd_file_path,
                     size,
                     target_resolution
                     )
    else:        
        logging.info('Already preprocessed this configuration. Loading now...')
        
    return h5py.File(test_sd_file_path, 'r')

# ===============================================================
# Main function that prepares a dataset from the raw challenge data to an hdf5 dataset.
# Extract the required files from their zipped directories
# ===============================================================
def prepare_test_sd_data(input_folder,
                 preprocessing_folder, 
                 test_sd_file_path,
                 size,
                 target_resolution
                 ):

    # ===============================
    # create a hdf5 file
    # ===============================
    hdf5_file = h5py.File(test_sd_file_path, "w")
    
    # ===============================
    # read all the patient folders from the base input folder
    # ===============================


    logging.info('Counting files and parsing meta data...')

    training_folder = input_folder + 'MICCAI_FeTS2021_TrainingData/'

    folder_list = []

    test_ids_sd = [8, 7, 6, 5, 4, 3, 2, 31, 32, 14, 22, 63, 62, 61, 60, 59, 58, 57, 56, 55]

    num_slices = 0

    for folder in os.listdir(training_folder):
        if not (folder.lower().endswith('.csv') or folder.lower().endswith('.md')):
            folder_path = os.path.join(training_folder, folder)  
            patient_id = int(folder.split('_')[-1])      
            if os.path.isdir(folder_path):
                if patient_id in test_ids_sd : 
                    folder_list.append(folder_path)

                    for _, _, fileList in os.walk(folder_path):
                        for filename in fileList:
                            
                                if filename.lower().endswith('t1.nii.gz'):
                                    image_t1, _, _ = utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices += image_t1.shape[2]
                                elif filename.lower().endswith('t1ce.nii.gz'):
                                    image_t1ce, _, _ = utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices += image_t1ce.shape[2]
                                elif filename.lower().endswith('t2.nii.gz'):
                                    image_t2, _, _= utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices += image_t2.shape[2]
                                elif filename.lower().endswith('flair.nii.gz'):
                                    image_flair, _, _ = utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices += image_flair.shape[2]



    # ===============================
    # Create datasets for images and labels
    # ===============================

    data = {}
    data['images'] = hdf5_file.create_dataset("images", list((size,size)) + [num_slices], dtype=np.float32)
    data['labels'] = hdf5_file.create_dataset("labels", list((size,size)) + [num_slices], dtype=np.uint8)

    #data = {}
    #num_slices = count_slices(folder_list, idx_start, idx_end)
    #data['images'] = hdf5_file.create_dataset("images", list((size,size)) + [num_slices], dtype=np.float32)
    #data['labels'] = hdf5_file.create_dataset("labels", list((size,size)) + [num_slices], dtype=np.uint8)
    
    # ===============================
    # initialize lists
    # ===============================    

    label_list = []
    image_list = []
    nx_list = []
    ny_list = []
    nz_list = []
    px_list = []
    py_list = []
    pz_list = []
    pat_names_list = []

    
    # ===============================        
    # ===============================        
    logging.info('Parsing image files')
    
        
    patient_counter = 0
    write_buffer = 0
    counter_from = 0

    for folder in folder_list:

        patient_counter += 1

        logging.info('================================')
        logging.info('Doing: %s' % folder)
        patname = folder.split('/')[-1]
        pat_names_list.append(patname)


        image_t1, _, image_t1_hdr = utils.load_nii(folder + f'/{patname}_t1.nii.gz')
        image_t1ce, _, image_t1ce_hdr = utils.load_nii(folder + f'/{patname}_t1ce.nii.gz')
        image_t2, _, image_t2_hdr = utils.load_nii(folder + f'/{patname}_t2.nii.gz')
        image_flair, _, image_flair_hdr = utils.load_nii(folder + f'/{patname}_flair.nii.gz')


        px_list.append(float(image_t1_hdr.get_zooms()[0]))
        py_list.append(float(image_t1_hdr.get_zooms()[1]))
        pz_list.append(float(image_t1_hdr.get_zooms()[2]))



        nifti_img_path = preprocessing_folder + '/Individual_NIFTI/' + patname + '/'
        if not os.path.exists(nifti_img_path):
            utils.makefolder(nifti_img_path)
        #utils.save_nii(img_path = nifti_img_path + '_img_t1.nii.gz', data = image_t1, affine = np.eye(4))
        #utils.save_nii(img_path = nifti_img_path + '_img_t1ce.nii.gz', data = image_t1ce, affine = np.eye(4))
        #utils.save_nii(img_path = nifti_img_path + '_img_t2.nii.gz', data = image_t2, affine = np.eye(4))
        #utils.save_nii(img_path = nifti_img_path + '_img_flair.nii.gz', data = image_flair, affine = np.eye(4))


        # ================================
        # do bias field correction
        # ================================
        input_img_t1 = nifti_img_path + patname + '_img_t1.nii.gz'
        output_img_t1 = nifti_img_path + patname + '_img_t1_n4.nii.gz'
        input_img_t1ce = nifti_img_path + patname + '_img_t1ce.nii.gz'
        output_img_t1ce = nifti_img_path + patname + '_img_t1ce_n4.nii.gz'
        input_img_t2 = nifti_img_path + patname + '_img_t2.nii.gz'
        output_img_t2 = nifti_img_path + patname + '_img_t2_n4.nii.gz'
        input_img_flair = nifti_img_path + patname + '_img_flair.nii.gz'
        output_img_flair = nifti_img_path + patname + '_img_flair_n4.nii.gz'


        # If bias corrected image does not exist, do it now
        for input_img, output_img in zip([input_img_t1, input_img_t1ce, input_img_t2, input_img_flair], [output_img_t1, output_img_t1ce, output_img_t2, output_img_flair]):
            if os.path.isfile(output_img):
                img = utils.load_nii(img_path = output_img)[0]
            else:
                subprocess.call(["/itet-stor/arismu/bmicdatasets_bmicnas01/Sharing/N4_th", input_img, output_img])
                img = utils.load_nii(img_path = output_img)[0]

            if input_img == input_img_t1:
                img_t1_n4 = img
                img_t1_n4 = utils.normalise_image(img_t1_n4, norm_type='div_by_max')
            elif input_img == input_img_t1ce:
                img_t1ce_n4 = img
                img_t1ce_n4 = utils.normalise_image(img_t1ce_n4, norm_type='div_by_max')
            elif input_img == input_img_t2:
                img_t2_n4 = img
                img_t2_n4 = utils.normalise_image(img_t2_n4, norm_type='div_by_max')
            elif input_img == input_img_flair:
                img_flair_n4 = img
                img_flair_n4 = utils.normalise_image(img_flair_n4, norm_type='div_by_max')

        image_n4 = np.concatenate((img_t1_n4, img_t1ce_n4), axis = 2)
        image_n4 = np.concatenate((image_n4, img_t2_n4), axis = 2)
        image_n4 = np.concatenate((image_n4, img_flair_n4), axis = 2)

        nx_list.append(image_n4.shape[0])
        ny_list.append(image_n4.shape[1])
        nz_list.append(image_n4.shape[2])
            

        # ================================    
        # read the labels
        # ================================   
        
        
        label, _, _ = utils.load_nii(folder + f'/{patname}_seg.nii.gz')

        label_temp = np.concatenate((label, label), axis = 2) #concatenate 3 times since all 4 image types share the same label
        label_temp = np.concatenate((label_temp, label), axis = 2)
        label = np.concatenate((label_temp, label), axis = 2)
        
        if not os.path.isfile(nifti_img_path + patname + '_lbl.nii.gz'):
            utils.save_nii(img_path = nifti_img_path + patname + '_lbl.nii.gz', data = label, affine = np.eye(4))

        ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################

        for zz in range(image_n4.shape[2]):

            #no rescaling needed since all images (SD & TDs) have same scale/dimensions already

            #image_cropped = crop_or_pad_slice_to_size(img_normalised[:, :, zz], size, size)
            #label_cropped = crop_or_pad_slice_to_size(label[:, :, zz], size, size)

            image_list.append(image_n4[:, :, zz])
            label_list.append(label[:, :, zz])

            write_buffer += 1

            if write_buffer >= MAX_WRITE_BUFFER:

                counter_to = counter_from + write_buffer

                _write_range_to_hdf5(data,
                                    image_list,
                                    label_list,
                                    counter_from,
                                    counter_to)
                
                _release_tmp_memory(image_list,
                                    label_list)

                # update counters 
                counter_from = counter_to
                write_buffer = 0


    #logging.info('Writing remaining data')
    #counter_to = counter_from + write_buffer

    #_write_range_to_hdf5(data, image_list, label_list, counter_from, counter_to)
   # _release_tmp_memory(image_list, label_list)


   
    hdf5_file.create_dataset('nx', data=np.asarray(nx_list, dtype=np.uint16))
    hdf5_file.create_dataset('ny', data=np.asarray(ny_list, dtype=np.uint16))
    hdf5_file.create_dataset('nz', data=np.asarray(nz_list, dtype=np.uint16))
    hdf5_file.create_dataset('px', data=np.asarray(px_list, dtype=np.float32))
    hdf5_file.create_dataset('py', data=np.asarray(py_list, dtype=np.float32))
    hdf5_file.create_dataset('pz', data=np.asarray(pz_list, dtype=np.float32))
    hdf5_file.create_dataset('patnames', data=np.asarray(pat_names_list, dtype="S20"))
    
    # After test train loop:
    logging.info('Test SD loop done')
    hdf5_file.close()







# ===============================
# TD1
# ===============================





def load_test_td1_data(input_folder,
              preproc_folder,
              size,
              target_resolution,
              force_overwrite = False):

    # ===============================
    # create the pre-processing folder, if it does not exist
    # ===============================
    utils.makefolder(preproc_folder)    
    
    # ===============================
    # file to create or directly read if it already exists
    # ===============================

    test_td1_file_name = 'data_test_td1.hdf5' 
    test_td1_file_path = os.path.join(preproc_folder, test_td1_file_name)
    
    # ===============================
    # if the images have not already been extracted, do so
    # ===============================
    if not os.path.exists(test_td1_file_path) or force_overwrite:
        
        logging.info('This configuration of protocol and data indices has not yet been preprocessed')
        logging.info('Preprocessing now...')
        prepare_test_td1_data(input_folder,
                     preproc_folder,
                     test_td1_file_path,
                     size,
                     target_resolution
                     )
    else:        
        logging.info('Already preprocessed this configuration. Loading now...')
        
    return h5py.File(test_td1_file_path, 'r')

# ===============================================================
# Main function that prepares a dataset from the raw challenge data to an hdf5 dataset.
# Extract the required files from their zipped directories
# ===============================================================
def prepare_test_td1_data(input_folder,
                 preprocessing_folder, 
                 test_td1_file_path,
                 size,
                 target_resolution
                 ):

    # ===============================
    # create a hdf5 file
    # ===============================
    hdf5_file = h5py.File(test_td1_file_path, "w")
    
    # ===============================
    # read all the patient folders from the base input folder
    # ===============================


    logging.info('Counting files and parsing meta data...')

    training_folder = input_folder + 'MICCAI_FeTS2021_TrainingData/'

    folder_list = []

    test_ids_td1= [351, 352, 353, 354, 355, 356, 357, 358, 359, 361, 367, 362, 363, 364, 365, 366, 350, 360, 349, 369, 347, 368, 336, 337, 
                      348, 339, 340, 338, 342, 343, 344, 345, 346, 341]
    num_slices = 0

    for folder in os.listdir(training_folder):
        if not (folder.lower().endswith('.csv') or folder.lower().endswith('.md')):
            folder_path = os.path.join(training_folder, folder)  
            patient_id = int(folder.split('_')[-1])      
            if os.path.isdir(folder_path):
                if patient_id in test_ids_td1 : 
                    folder_list.append(folder_path)

                    for _, _, fileList in os.walk(folder_path):
                        for filename in fileList:
                            
                                if filename.lower().endswith('t1.nii.gz'):
                                    image_t1, _, _ = utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices += image_t1.shape[2]
                                elif filename.lower().endswith('t1ce.nii.gz'):
                                    image_t1ce, _, _ = utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices += image_t1ce.shape[2]
                                elif filename.lower().endswith('t2.nii.gz'):
                                    image_t2, _, _= utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices += image_t2.shape[2]
                                elif filename.lower().endswith('flair.nii.gz'):
                                    image_flair, _, _ = utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices += image_flair.shape[2]



    # ===============================
    # Create datasets for images and labels
    # ===============================

    data = {}
    data['images'] = hdf5_file.create_dataset("images", list((size,size)) + [num_slices], dtype=np.float32)
    data['labels'] = hdf5_file.create_dataset("labels", list((size,size)) + [num_slices], dtype=np.uint8)

    #data = {}
    #num_slices = count_slices(folder_list, idx_start, idx_end)
    #data['images'] = hdf5_file.create_dataset("images", list((size,size)) + [num_slices], dtype=np.float32)
    #data['labels'] = hdf5_file.create_dataset("labels", list((size,size)) + [num_slices], dtype=np.uint8)
    
    # ===============================
    # initialize lists
    # ===============================    

    label_list = []
    image_list = []
    nx_list = []
    ny_list = []
    nz_list = []
    px_list = []
    py_list = []
    pz_list = []
    pat_names_list = []

    
    # ===============================        
    # ===============================        
    logging.info('Parsing image files')
    
        
    patient_counter = 0
    write_buffer = 0
    counter_from = 0

    for folder in folder_list:

        patient_counter += 1

        logging.info('================================')
        logging.info('Doing: %s' % folder)
        patname = folder.split('/')[-1]
        pat_names_list.append(patname)


        image_t1, _, image_t1_hdr = utils.load_nii(folder + f'/{patname}_t1.nii.gz')
        image_t1ce, _, image_t1ce_hdr = utils.load_nii(folder + f'/{patname}_t1ce.nii.gz')
        image_t2, _, image_t2_hdr = utils.load_nii(folder + f'/{patname}_t2.nii.gz')
        image_flair, _, image_flair_hdr = utils.load_nii(folder + f'/{patname}_flair.nii.gz')


        px_list.append(float(image_t1_hdr.get_zooms()[0]))
        py_list.append(float(image_t1_hdr.get_zooms()[1]))
        pz_list.append(float(image_t1_hdr.get_zooms()[2]))



        nifti_img_path = preprocessing_folder + '/Individual_NIFTI/' + patname + '/'
        if not os.path.exists(nifti_img_path):
            utils.makefolder(nifti_img_path)
        #utils.save_nii(img_path = nifti_img_path + '_img_t1.nii.gz', data = image_t1, affine = np.eye(4))
        #utils.save_nii(img_path = nifti_img_path + '_img_t1ce.nii.gz', data = image_t1ce, affine = np.eye(4))
        #utils.save_nii(img_path = nifti_img_path + '_img_t2.nii.gz', data = image_t2, affine = np.eye(4))
        #utils.save_nii(img_path = nifti_img_path + '_img_flair.nii.gz', data = image_flair, affine = np.eye(4))


        # ================================
        # do bias field correction
        # ================================
        input_img_t1 = nifti_img_path + patname + '_img_t1.nii.gz'
        output_img_t1 = nifti_img_path + patname + '_img_t1_n4.nii.gz'
        input_img_t1ce = nifti_img_path + patname + '_img_t1ce.nii.gz'
        output_img_t1ce = nifti_img_path + patname + '_img_t1ce_n4.nii.gz'
        input_img_t2 = nifti_img_path + patname + '_img_t2.nii.gz'
        output_img_t2 = nifti_img_path + patname + '_img_t2_n4.nii.gz'
        input_img_flair = nifti_img_path + patname + '_img_flair.nii.gz'
        output_img_flair = nifti_img_path + patname + '_img_flair_n4.nii.gz'


        # If bias corrected image does not exist, do it now
        for input_img, output_img in zip([input_img_t1, input_img_t1ce, input_img_t2, input_img_flair], [output_img_t1, output_img_t1ce, output_img_t2, output_img_flair]):
            if os.path.isfile(output_img):
                img = utils.load_nii(img_path = output_img)[0]
            else:
                subprocess.call(["/itet-stor/arismu/bmicdatasets_bmicnas01/Sharing/N4_th", input_img, output_img])
                img = utils.load_nii(img_path = output_img)[0]

            if input_img == input_img_t1:
                img_t1_n4 = img
                img_t1_n4 = utils.normalise_image(img_t1_n4, norm_type='div_by_max')
            elif input_img == input_img_t1ce:
                img_t1ce_n4 = img
                img_t1ce_n4 = utils.normalise_image(img_t1ce_n4, norm_type='div_by_max')
            elif input_img == input_img_t2:
                img_t2_n4 = img
                img_t2_n4 = utils.normalise_image(img_t2_n4, norm_type='div_by_max')
            elif input_img == input_img_flair:
                img_flair_n4 = img
                img_flair_n4 = utils.normalise_image(img_flair_n4, norm_type='div_by_max')

        image_n4 = np.concatenate((img_t1_n4, img_t1ce_n4), axis = 2)
        image_n4 = np.concatenate((image_n4, img_t2_n4), axis = 2)
        image_n4 = np.concatenate((image_n4, img_flair_n4), axis = 2)

        nx_list.append(image_n4.shape[0])
        ny_list.append(image_n4.shape[1])
        nz_list.append(image_n4.shape[2])
            

        # ================================    
        # read the labels
        # ================================   
        
        
        label, _, _ = utils.load_nii(folder + f'/{patname}_seg.nii.gz')

        label_temp = np.concatenate((label, label), axis = 2) #concatenate 3 times since all 4 image types share the same label
        label_temp = np.concatenate((label_temp, label), axis = 2)
        label = np.concatenate((label_temp, label), axis = 2)
        
        if not os.path.isfile(nifti_img_path + patname + '_lbl.nii.gz'):
            utils.save_nii(img_path = nifti_img_path + patname + '_lbl.nii.gz', data = label, affine = np.eye(4))

        ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################

        for zz in range(image_n4.shape[2]):

            #no rescaling needed since all images (SD & TDs) have same scale/dimensions already

            #image_cropped = crop_or_pad_slice_to_size(img_normalised[:, :, zz], size, size)
            #label_cropped = crop_or_pad_slice_to_size(label[:, :, zz], size, size)

            image_list.append(image_n4[:, :, zz])
            label_list.append(label[:, :, zz])

            write_buffer += 1

            if write_buffer >= MAX_WRITE_BUFFER:

                counter_to = counter_from + write_buffer

                _write_range_to_hdf5(data,
                                    image_list,
                                    label_list,
                                    counter_from,
                                    counter_to)
                
                _release_tmp_memory(image_list,
                                    label_list)

                # update counters 
                counter_from = counter_to
                write_buffer = 0


    #logging.info('Writing remaining data')
    #counter_to = counter_from + write_buffer

    #_write_range_to_hdf5(data, image_list, label_list, counter_from, counter_to)
   # _release_tmp_memory(image_list, label_list)


   
    hdf5_file.create_dataset('nx', data=np.asarray(nx_list, dtype=np.uint16))
    hdf5_file.create_dataset('ny', data=np.asarray(ny_list, dtype=np.uint16))
    hdf5_file.create_dataset('nz', data=np.asarray(nz_list, dtype=np.uint16))
    hdf5_file.create_dataset('px', data=np.asarray(px_list, dtype=np.float32))
    hdf5_file.create_dataset('py', data=np.asarray(py_list, dtype=np.float32))
    hdf5_file.create_dataset('pz', data=np.asarray(pz_list, dtype=np.float32))
    hdf5_file.create_dataset('patnames', data=np.asarray(pat_names_list, dtype="S20"))
    
    # After test train loop:
    logging.info('Test TD1 loop done')
    hdf5_file.close()







# ===============================
# TD2
# ===============================

def load_test_td2_data(input_folder,
              preproc_folder,
              size,
              target_resolution,
              force_overwrite = False):

    # ===============================
    # create the pre-processing folder, if it does not exist
    # ===============================
    utils.makefolder(preproc_folder)    
    
    # ===============================
    # file to create or directly read if it already exists
    # ===============================

    test_td2_file_name = 'data_test_td2.hdf5' 
    test_td2_file_path = os.path.join(preproc_folder, test_td2_file_name)
    
    # ===============================
    # if the images have not already been extracted, do so
    # ===============================
    if not os.path.exists(test_td2_file_path) or force_overwrite:
        
        logging.info('This configuration of protocol and data indices has not yet been preprocessed')
        logging.info('Preprocessing now...')
        prepare_test_td2_data(input_folder,
                     preproc_folder,
                     test_td2_file_path,
                     size,
                     target_resolution
                     )
    else:        
        logging.info('Already preprocessed this configuration. Loading now...')
        
    return h5py.File(test_td2_file_path, 'r')

# ===============================================================
# Main function that prepares a dataset from the raw challenge data to an hdf5 dataset.
# Extract the required files from their zipped directories
# ===============================================================
def prepare_test_td2_data(input_folder,
                 preprocessing_folder, 
                 test_td2_file_path,
                 size,
                 target_resolution
                 ):

    # ===============================
    # create a hdf5 file
    # ===============================
    hdf5_file = h5py.File(test_td2_file_path, "w")
    
    # ===============================
    # read all the patient folders from the base input folder
    # ===============================


    logging.info('Counting files and parsing meta data...')

    training_folder = input_folder + 'MICCAI_FeTS2021_TrainingData/'

    folder_list = []

    test_ids_td2 = [204, 199, 200, 201, 202, 203, 206, 211, 208, 209, 210, 198, 212, 213, 207, 197, 205, 195, 181, 182, 183, 184, 185, 187, 188, 
                186, 189, 190, 191, 192, 193, 194, 180, 196]

    num_slices = 0

    for folder in os.listdir(training_folder):
        if not (folder.lower().endswith('.csv') or folder.lower().endswith('.md')):
            folder_path = os.path.join(training_folder, folder)  
            patient_id = int(folder.split('_')[-1])      
            if os.path.isdir(folder_path):
                if patient_id in test_ids_td2 : 
                    folder_list.append(folder_path)

                    for _, _, fileList in os.walk(folder_path):
                        for filename in fileList:
                            
                                if filename.lower().endswith('t1.nii.gz'):
                                    image_t1, _, _ = utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices += image_t1.shape[2]
                                elif filename.lower().endswith('t1ce.nii.gz'):
                                    image_t1ce, _, _ = utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices += image_t1ce.shape[2]
                                elif filename.lower().endswith('t2.nii.gz'):
                                    image_t2, _, _= utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices += image_t2.shape[2]
                                elif filename.lower().endswith('flair.nii.gz'):
                                    image_flair, _, _ = utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices += image_flair.shape[2]



    # ===============================
    # Create datasets for images and labels
    # ===============================

    data = {}
    data['images'] = hdf5_file.create_dataset("images", list((size,size)) + [num_slices], dtype=np.float32)
    data['labels'] = hdf5_file.create_dataset("labels", list((size,size)) + [num_slices], dtype=np.uint8)

    #data = {}
    #num_slices = count_slices(folder_list, idx_start, idx_end)
    #data['images'] = hdf5_file.create_dataset("images", list((size,size)) + [num_slices], dtype=np.float32)
    #data['labels'] = hdf5_file.create_dataset("labels", list((size,size)) + [num_slices], dtype=np.uint8)
    
    # ===============================
    # initialize lists
    # ===============================    

    label_list = []
    image_list = []
    nx_list = []
    ny_list = []
    nz_list = []
    px_list = []
    py_list = []
    pz_list = []
    pat_names_list = []

    
    # ===============================        
    # ===============================        
    logging.info('Parsing image files')
    
        
    patient_counter = 0
    write_buffer = 0
    counter_from = 0

    for folder in folder_list:

        patient_counter += 1

        logging.info('================================')
        logging.info('Doing: %s' % folder)
        patname = folder.split('/')[-1]
        pat_names_list.append(patname)


        image_t1, _, image_t1_hdr = utils.load_nii(folder + f'/{patname}_t1.nii.gz')
        image_t1ce, _, image_t1ce_hdr = utils.load_nii(folder + f'/{patname}_t1ce.nii.gz')
        image_t2, _, image_t2_hdr = utils.load_nii(folder + f'/{patname}_t2.nii.gz')
        image_flair, _, image_flair_hdr = utils.load_nii(folder + f'/{patname}_flair.nii.gz')


        px_list.append(float(image_t1_hdr.get_zooms()[0]))
        py_list.append(float(image_t1_hdr.get_zooms()[1]))
        pz_list.append(float(image_t1_hdr.get_zooms()[2]))



        nifti_img_path = preprocessing_folder + '/Individual_NIFTI/' + patname + '/'
        if not os.path.exists(nifti_img_path):
            utils.makefolder(nifti_img_path)
        #utils.save_nii(img_path = nifti_img_path + '_img_t1.nii.gz', data = image_t1, affine = np.eye(4))
        #utils.save_nii(img_path = nifti_img_path + '_img_t1ce.nii.gz', data = image_t1ce, affine = np.eye(4))
        #utils.save_nii(img_path = nifti_img_path + '_img_t2.nii.gz', data = image_t2, affine = np.eye(4))
        #utils.save_nii(img_path = nifti_img_path + '_img_flair.nii.gz', data = image_flair, affine = np.eye(4))


        # ================================
        # do bias field correction
        # ================================
        input_img_t1 = nifti_img_path + patname + '_img_t1.nii.gz'
        output_img_t1 = nifti_img_path + patname + '_img_t1_n4.nii.gz'
        input_img_t1ce = nifti_img_path + patname + '_img_t1ce.nii.gz'
        output_img_t1ce = nifti_img_path + patname + '_img_t1ce_n4.nii.gz'
        input_img_t2 = nifti_img_path + patname + '_img_t2.nii.gz'
        output_img_t2 = nifti_img_path + patname + '_img_t2_n4.nii.gz'
        input_img_flair = nifti_img_path + patname + '_img_flair.nii.gz'
        output_img_flair = nifti_img_path + patname + '_img_flair_n4.nii.gz'


        # If bias corrected image does not exist, do it now
        for input_img, output_img in zip([input_img_t1, input_img_t1ce, input_img_t2, input_img_flair], [output_img_t1, output_img_t1ce, output_img_t2, output_img_flair]):
            if os.path.isfile(output_img):
                img = utils.load_nii(img_path = output_img)[0]
            else:
                subprocess.call(["/itet-stor/arismu/bmicdatasets_bmicnas01/Sharing/N4_th", input_img, output_img])
                img = utils.load_nii(img_path = output_img)[0]

            if input_img == input_img_t1:
                img_t1_n4 = img
                img_t1_n4 = utils.normalise_image(img_t1_n4, norm_type='div_by_max')
            elif input_img == input_img_t1ce:
                img_t1ce_n4 = img
                img_t1ce_n4 = utils.normalise_image(img_t1ce_n4, norm_type='div_by_max')
            elif input_img == input_img_t2:
                img_t2_n4 = img
                img_t2_n4 = utils.normalise_image(img_t2_n4, norm_type='div_by_max')
            elif input_img == input_img_flair:
                img_flair_n4 = img
                img_flair_n4 = utils.normalise_image(img_flair_n4, norm_type='div_by_max')

        image_n4 = np.concatenate((img_t1_n4, img_t1ce_n4), axis = 2)
        image_n4 = np.concatenate((image_n4, img_t2_n4), axis = 2)
        image_n4 = np.concatenate((image_n4, img_flair_n4), axis = 2)

        nx_list.append(image_n4.shape[0])
        ny_list.append(image_n4.shape[1])
        nz_list.append(image_n4.shape[2])
            

        # ================================    
        # read the labels
        # ================================   
        
        
        label, _, _ = utils.load_nii(folder + f'/{patname}_seg.nii.gz')

        label_temp = np.concatenate((label, label), axis = 2) #concatenate 3 times since all 4 image types share the same label
        label_temp = np.concatenate((label_temp, label), axis = 2)
        label = np.concatenate((label_temp, label), axis = 2)
        
        if not os.path.isfile(nifti_img_path + patname + '_lbl.nii.gz'):
            utils.save_nii(img_path = nifti_img_path + patname + '_lbl.nii.gz', data = label, affine = np.eye(4))

        ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################

        for zz in range(image_n4.shape[2]):

            #no rescaling needed since all images (SD & TDs) have same scale/dimensions already

            #image_cropped = crop_or_pad_slice_to_size(img_normalised[:, :, zz], size, size)
            #label_cropped = crop_or_pad_slice_to_size(label[:, :, zz], size, size)

            image_list.append(image_n4[:, :, zz])
            label_list.append(label[:, :, zz])

            write_buffer += 1

            if write_buffer >= MAX_WRITE_BUFFER:

                counter_to = counter_from + write_buffer

                _write_range_to_hdf5(data,
                                    image_list,
                                    label_list,
                                    counter_from,
                                    counter_to)
                
                _release_tmp_memory(image_list,
                                    label_list)

                # update counters 
                counter_from = counter_to
                write_buffer = 0


    #logging.info('Writing remaining data')
    #counter_to = counter_from + write_buffer

    #_write_range_to_hdf5(data, image_list, label_list, counter_from, counter_to)
   # _release_tmp_memory(image_list, label_list)


   
    hdf5_file.create_dataset('nx', data=np.asarray(nx_list, dtype=np.uint16))
    hdf5_file.create_dataset('ny', data=np.asarray(ny_list, dtype=np.uint16))
    hdf5_file.create_dataset('nz', data=np.asarray(nz_list, dtype=np.uint16))
    hdf5_file.create_dataset('px', data=np.asarray(px_list, dtype=np.float32))
    hdf5_file.create_dataset('py', data=np.asarray(py_list, dtype=np.float32))
    hdf5_file.create_dataset('pz', data=np.asarray(pz_list, dtype=np.float32))
    hdf5_file.create_dataset('patnames', data=np.asarray(pat_names_list, dtype="S20"))
    
    # After test train loop:
    logging.info('Test TD2 loop done')
    hdf5_file.close()





# ===============================
# Training Data
# ===============================

def load_training_data(input_folder,
              preproc_folder,
              size,
              target_resolution,
              force_overwrite = False):

    # ===============================
    # create the pre-processing folder, if it does not exist
    # ===============================
    utils.makefolder(preproc_folder)    
    
    # ===============================
    # file to create or directly read if it already exists
    # ===============================

    train_file_name_part1 = 'data_training_part1.hdf5' 
    train_file_path_part1 = os.path.join(preproc_folder, train_file_name_part1)

    train_file_name_part2 = 'data_training_part2.hdf5' 
    train_file_path_part2 = os.path.join(preproc_folder, train_file_name_part2)

    train_file_name_part3 = 'data_training_part3.hdf5' 
    train_file_path_part3 = os.path.join(preproc_folder, train_file_name_part3)


    
    # ===============================
    # if the images have not already been extracted, do so
    # ===============================
    #if not os.path.exists(train_file_path_part1 or train_file_path_part2 or train_file_path_part3) or force_overwrite:
    if not os.path.exists(train_file_path_part1 or train_file_path_part2 or train_file_path_part3) or force_overwrite:

        logging.info('This configuration of protocol and data indices has not yet been preprocessed')
        logging.info('Preprocessing now...')
        prepare_training_data(input_folder,
                     preproc_folder,
                     train_file_path_part1,
                     train_file_path_part2,
                     train_file_path_part3,
                     size,
                     target_resolution
                     )
    else:        
        logging.info('Already preprocessed this configuration. Loading now...')
        
    return h5py.File(train_file_path_part1, 'r'), h5py.File(train_file_path_part2, 'r'), h5py.File(train_file_path_part3, 'r')





def prepare_training_data(input_folder,
                 preprocessing_folder, 
                 train_file_path_part1,
                 train_file_path_part2,
                 train_file_path_part3,
                 size,
                 target_resolution
                 ):

    # ===============================
    # create a hdf5 file
    # ===============================
    hdf5_file_part1 = h5py.File(train_file_path_part1, "w")
    
    

    
    # ===============================
    # read all the patient folders from the base input folder
    # ===============================


    logging.info('Counting files and parsing meta data...')

    training_folder = input_folder + 'MICCAI_FeTS2021_TrainingData/'

    folder_list_part1 = []
    folder_list_part2 = []
    folder_list_part3 = []

    training_ids_part1 = [1, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 83, 97, 82, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68] 
    training_ids_part2 = [81, 98, 99, 100, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109,
                          108, 107, 106, 105] 
    training_ids_part3 = [104, 103, 102, 101, 67, 66, 84, 64, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 65, 13, 12, 11, 10, 9]
    #validation_ids = [54, 53, 51, 50, 49, 52, 47, 35, 36, 37, 38, 48, 40, 41, 39, 43, 44, 45, 46, 42]

    num_slices_part1 = 0
    num_slices_part2 = 0
    num_slices_part3 = 0


    for folder in os.listdir(training_folder):
        if not (folder.lower().endswith('.csv') or folder.lower().endswith('.md')):
            folder_path = os.path.join(training_folder, folder)  
            patient_id = int(folder.split('_')[-1])      
            if os.path.isdir(folder_path):
                if patient_id in training_ids_part1: 
                    folder_list_part1.append(folder_path)

                    for _, _, fileList in os.walk(folder_path):
                        for filename in fileList:
                                if filename.lower().endswith('t1.nii.gz'):
                                    image_t1, _, _ = utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices_part1 += image_t1.shape[2]
                                elif filename.lower().endswith('t1ce.nii.gz'):
                                    image_t1ce, _, _ = utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices_part1 += image_t1ce.shape[2]
                                elif filename.lower().endswith('t2.nii.gz'):
                                    image_t2, _, _= utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices_part1 += image_t2.shape[2]
                                elif filename.lower().endswith('flair.nii.gz'):
                                    image_flair, _, _ = utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices_part1 += image_flair.shape[2]
                    
                elif patient_id in training_ids_part2:
                    folder_list_part2.append(folder_path)
                    for _, _, fileList in os.walk(folder_path):
                        for filename in fileList:
                                if filename.lower().endswith('t1.nii.gz'):
                                    image_t1, _, _ = utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices_part2 += image_t1.shape[2]
                                elif filename.lower().endswith('t1ce.nii.gz'):
                                    image_t1ce, _, _ = utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices_part2 += image_t1ce.shape[2]
                                elif filename.lower().endswith('t2.nii.gz'):
                                    image_t2, _, _= utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices_part2 += image_t2.shape[2]
                                elif filename.lower().endswith('flair.nii.gz'):
                                    image_flair, _, _ = utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices_part2 += image_flair.shape[2]

                elif patient_id in training_ids_part3:
                    folder_list_part3.append(folder_path)
                    for _, _, fileList in os.walk(folder_path):
                        for filename in fileList:
                                if filename.lower().endswith('t1.nii.gz'):
                                    image_t1, _, _ = utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices_part3 += image_t1.shape[2]
                                elif filename.lower().endswith('t1ce.nii.gz'):
                                    image_t1ce, _, _ = utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices_part3 += image_t1ce.shape[2]
                                elif filename.lower().endswith('t2.nii.gz'):
                                    image_t2, _, _= utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices_part3 += image_t2.shape[2]
                                elif filename.lower().endswith('flair.nii.gz'):
                                    image_flair, _, _ = utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices_part3 += image_flair.shape[2]
        

    # ===============================
    # Create datasets for images and labels
    # ===============================

    data_part1 = {}

    data_part1['images'] = hdf5_file_part1.create_dataset("images", list((size,size)) + [num_slices_part1], dtype=np.float32)
    data_part1['labels'] = hdf5_file_part1.create_dataset("labels", list((size,size)) + [num_slices_part1], dtype=np.uint8)

    

    #data = {}
    #num_slices = count_slices(folder_list, idx_start, idx_end)
    #data['images'] = hdf5_file.create_dataset("images", list((size,size)) + [num_slices], dtype=np.float32)
    #data['labels'] = hdf5_file.create_dataset("labels", list((size,size)) + [num_slices], dtype=np.uint8)
    
    # ===============================
    # initialize lists
    # ===============================    

    label_list_part1 = []
    image_list_part1 = []
    nx_list_part1 = []
    ny_list_part1 = []
    nz_list_part1 = []
    px_list_part1 = []
    py_list_part1 = []
    pz_list_part1= []
    pat_names_list_part1 = []

    label_list_part2 = []
    image_list_part2 = []
    nx_list_part2 = []
    ny_list_part2 = []
    nz_list_part2 = []
    px_list_part2 = []
    py_list_part2 = []
    pz_list_part2 = []
    pat_names_list_part2 = []

    label_list_part3 = []
    image_list_part3 = []
    nx_list_part3 = []
    ny_list_part3 = []
    nz_list_part3 = []
    px_list_part3 = []
    py_list_part3 = []
    pz_list_part3= []
    pat_names_list_part3 = []

    # ===============================        
    # ===============================        
    logging.info('Parsing image files')
    
        
    patient_counter = 0
    write_buffer = 0
    counter_from = 0

    for folder in folder_list_part1:

        patient_counter += 1

        logging.info('================================')
        logging.info('Doing: %s' % folder)
        patname = folder.split('/')[-1]
        pat_names_list_part1.append(patname)


        image_t1, _, image_t1_hdr = utils.load_nii(folder + f'/{patname}_t1.nii.gz')
        image_t1ce, _, image_t1ce_hdr = utils.load_nii(folder + f'/{patname}_t1ce.nii.gz')
        image_t2, _, image_t2_hdr = utils.load_nii(folder + f'/{patname}_t2.nii.gz')
        image_flair, _, image_flair_hdr = utils.load_nii(folder + f'/{patname}_flair.nii.gz')


        px_list_part1.append(float(image_t1_hdr.get_zooms()[0]))
        py_list_part1.append(float(image_t1_hdr.get_zooms()[1]))
        pz_list_part1.append(float(image_t1_hdr.get_zooms()[2]))



        nifti_img_path = preprocessing_folder + '/Individual_NIFTI/' + patname + '/'
        if not os.path.exists(nifti_img_path):
            utils.makefolder(nifti_img_path)
        #utils.save_nii(img_path = nifti_img_path + '_img_t1.nii.gz', data = image_t1, affine = np.eye(4))
        #utils.save_nii(img_path = nifti_img_path + '_img_t1ce.nii.gz', data = image_t1ce, affine = np.eye(4))
        #utils.save_nii(img_path = nifti_img_path + '_img_t2.nii.gz', data = image_t2, affine = np.eye(4))
        #utils.save_nii(img_path = nifti_img_path + '_img_flair.nii.gz', data = image_flair, affine = np.eye(4))


        # ================================
        # do bias field correction
        # ================================
        input_img_t1 = nifti_img_path + patname + '_img_t1.nii.gz'
        output_img_t1 = nifti_img_path + patname + '_img_t1_n4.nii.gz'
        input_img_t1ce = nifti_img_path + patname + '_img_t1ce.nii.gz'
        output_img_t1ce = nifti_img_path + patname + '_img_t1ce_n4.nii.gz'
        input_img_t2 = nifti_img_path + patname + '_img_t2.nii.gz'
        output_img_t2 = nifti_img_path + patname + '_img_t2_n4.nii.gz'
        input_img_flair = nifti_img_path + patname + '_img_flair.nii.gz'
        output_img_flair = nifti_img_path + patname + '_img_flair_n4.nii.gz'


        # If bias corrected image does not exist, do it now
        for input_img, output_img in zip([input_img_t1, input_img_t1ce, input_img_t2, input_img_flair], [output_img_t1, output_img_t1ce, output_img_t2, output_img_flair]):
            if os.path.isfile(output_img):
                img = utils.load_nii(img_path = output_img)[0]
            else:
                subprocess.call(["/itet-stor/arismu/bmicdatasets_bmicnas01/Sharing/N4_th", input_img, output_img])
                img = utils.load_nii(img_path = output_img)[0]

            if input_img == input_img_t1:
                img_t1_n4 = img
                img_t1_n4 = utils.normalise_image(img_t1_n4, norm_type='div_by_max')
            elif input_img == input_img_t1ce:
                img_t1ce_n4 = img
                img_t1ce_n4 = utils.normalise_image(img_t1ce_n4, norm_type='div_by_max')
            elif input_img == input_img_t2:
                img_t2_n4 = img
                img_t2_n4 = utils.normalise_image(img_t2_n4, norm_type='div_by_max')
            elif input_img == input_img_flair:
                img_flair_n4 = img
                img_flair_n4 = utils.normalise_image(img_flair_n4, norm_type='div_by_max')

        image_n4 = np.concatenate((img_t1_n4, img_t1ce_n4), axis = 2)
        image_n4 = np.concatenate((image_n4, img_t2_n4), axis = 2)
        image_n4 = np.concatenate((image_n4, img_flair_n4), axis = 2)

        nx_list_part1.append(image_n4.shape[0])
        ny_list_part1.append(image_n4.shape[1])
        nz_list_part1.append(image_n4.shape[2])
            

        # ================================    
        # read the labels
        # ================================   
        
        
        label, _, _ = utils.load_nii(folder + f'/{patname}_seg.nii.gz')

        label_temp = np.concatenate((label, label), axis = 2) #concatenate 3 times since all 4 image types share the same label
        label_temp = np.concatenate((label_temp, label), axis = 2)
        label = np.concatenate((label_temp, label), axis = 2)
        
        if not os.path.isfile(nifti_img_path + patname + '_lbl.nii.gz'):
            utils.save_nii(img_path = nifti_img_path + patname + '_lbl.nii.gz', data = label, affine = np.eye(4))

        ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################

        for zz in range(image_n4.shape[2]):

            #no rescaling needed since all images (SD & TDs) have same scale/dimensions already

            #image_cropped = crop_or_pad_slice_to_size(img_normalised[:, :, zz], size, size)
            #label_cropped = crop_or_pad_slice_to_size(label[:, :, zz], size, size)

            image_list_part1.append(image_n4[:, :, zz])
            label_list_part1.append(label[:, :, zz])

            write_buffer += 1

            if write_buffer >= MAX_WRITE_BUFFER:

                counter_to = counter_from + write_buffer

                _write_range_to_hdf5(data_part1,
                                    image_list_part1,
                                    label_list_part1,
                                    counter_from,
                                    counter_to)
                
                _release_tmp_memory(image_list_part1,
                                    label_list_part1)

                # update counters 
                counter_from = counter_to
                write_buffer = 0


    #logging.info('Writing remaining data')
    #counter_to = counter_from + write_buffer

    #_write_range_to_hdf5(data, image_list, label_list, counter_from, counter_to)
   # _release_tmp_memory(image_list, label_list)


   
    hdf5_file_part1.create_dataset('nx', data=np.asarray(nx_list_part1, dtype=np.uint16))
    hdf5_file_part1.create_dataset('ny', data=np.asarray(ny_list_part1, dtype=np.uint16))
    hdf5_file_part1.create_dataset('nz', data=np.asarray(nz_list_part1, dtype=np.uint16))
    hdf5_file_part1.create_dataset('px', data=np.asarray(px_list_part1, dtype=np.float32))
    hdf5_file_part1.create_dataset('py', data=np.asarray(py_list_part1, dtype=np.float32))
    hdf5_file_part1.create_dataset('pz', data=np.asarray(pz_list_part1, dtype=np.float32))
    hdf5_file_part1.create_dataset('patnames', data=np.asarray(pat_names_list_part1, dtype="S20"))
    
    # After test train loop:
    logging.info('Training part1 loop done')
    hdf5_file_part1.close()


    hdf5_file_part2 = h5py.File(train_file_path_part2, "w")
    data_part2 = {}
    data_part2['images'] = hdf5_file_part2.create_dataset("images", list((size,size)) + [num_slices_part2], dtype=np.float32)
    data_part2['labels'] = hdf5_file_part2.create_dataset("labels", list((size,size)) + [num_slices_part2], dtype=np.uint8)


    for folder in folder_list_part2:

        patient_counter += 1

        logging.info('================================')
        logging.info('Doing: %s' % folder)
        patname = folder.split('/')[-1]
        pat_names_list_part2.append(patname)


        image_t1, _, image_t1_hdr = utils.load_nii(folder + f'/{patname}_t1.nii.gz')
        image_t1ce, _, image_t1ce_hdr = utils.load_nii(folder + f'/{patname}_t1ce.nii.gz')
        image_t2, _, image_t2_hdr = utils.load_nii(folder + f'/{patname}_t2.nii.gz')
        image_flair, _, image_flair_hdr = utils.load_nii(folder + f'/{patname}_flair.nii.gz')


        px_list_part2.append(float(image_t1_hdr.get_zooms()[0]))
        py_list_part2.append(float(image_t1_hdr.get_zooms()[1]))
        pz_list_part2.append(float(image_t1_hdr.get_zooms()[2]))



        nifti_img_path = preprocessing_folder + '/Individual_NIFTI/' + patname + '/'
        if not os.path.exists(nifti_img_path):
            utils.makefolder(nifti_img_path)
        #utils.save_nii(img_path = nifti_img_path + '_img_t1.nii.gz', data = image_t1, affine = np.eye(4))
        #utils.save_nii(img_path = nifti_img_path + '_img_t1ce.nii.gz', data = image_t1ce, affine = np.eye(4))
        #utils.save_nii(img_path = nifti_img_path + '_img_t2.nii.gz', data = image_t2, affine = np.eye(4))
        #utils.save_nii(img_path = nifti_img_path + '_img_flair.nii.gz', data = image_flair, affine = np.eye(4))


        # ================================
        # do bias field correction
        # ================================
        input_img_t1 = nifti_img_path + patname + '_img_t1.nii.gz'
        output_img_t1 = nifti_img_path + patname + '_img_t1_n4.nii.gz'
        input_img_t1ce = nifti_img_path + patname + '_img_t1ce.nii.gz'
        output_img_t1ce = nifti_img_path + patname + '_img_t1ce_n4.nii.gz'
        input_img_t2 = nifti_img_path + patname + '_img_t2.nii.gz'
        output_img_t2 = nifti_img_path + patname + '_img_t2_n4.nii.gz'
        input_img_flair = nifti_img_path + patname + '_img_flair.nii.gz'
        output_img_flair = nifti_img_path + patname + '_img_flair_n4.nii.gz'


        # If bias corrected image does not exist, do it now
        for input_img, output_img in zip([input_img_t1, input_img_t1ce, input_img_t2, input_img_flair], [output_img_t1, output_img_t1ce, output_img_t2, output_img_flair]):
            if os.path.isfile(output_img):
                img = utils.load_nii(img_path = output_img)[0]
            else:
                subprocess.call(["/itet-stor/arismu/bmicdatasets_bmicnas01/Sharing/N4_th", input_img, output_img])
                img = utils.load_nii(img_path = output_img)[0]

            if input_img == input_img_t1:
                img_t1_n4 = img
                img_t1_n4 = utils.normalise_image(img_t1_n4, norm_type='div_by_max')
            elif input_img == input_img_t1ce:
                img_t1ce_n4 = img
                img_t1ce_n4 = utils.normalise_image(img_t1ce_n4, norm_type='div_by_max')
            elif input_img == input_img_t2:
                img_t2_n4 = img
                img_t2_n4 = utils.normalise_image(img_t2_n4, norm_type='div_by_max')
            elif input_img == input_img_flair:
                img_flair_n4 = img
                img_flair_n4 = utils.normalise_image(img_flair_n4, norm_type='div_by_max')

        image_n4 = np.concatenate((img_t1_n4, img_t1ce_n4), axis = 2)
        image_n4 = np.concatenate((image_n4, img_t2_n4), axis = 2)
        image_n4 = np.concatenate((image_n4, img_flair_n4), axis = 2)

        nx_list_part2.append(image_n4.shape[0])
        ny_list_part2.append(image_n4.shape[1])
        nz_list_part2.append(image_n4.shape[2])
            


        # ================================    
        # read the labels
        # ================================   
        
        
        label, _, _ = utils.load_nii(folder + f'/{patname}_seg.nii.gz')

        label_temp = np.concatenate((label, label), axis = 2) #concatenate 3 times since all 4 image types share the same label
        label_temp = np.concatenate((label_temp, label), axis = 2)
        label = np.concatenate((label_temp, label), axis = 2)
        
        if not os.path.isfile(nifti_img_path + patname + '_lbl.nii.gz'):
            utils.save_nii(img_path = nifti_img_path + patname + '_lbl.nii.gz', data = label, affine = np.eye(4))

        ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################

        for zz in range(image_n4.shape[2]):

            #no rescaling needed since all images (SD & TDs) have same scale/dimensions already

            #image_cropped = crop_or_pad_slice_to_size(img_normalised[:, :, zz], size, size)
            #label_cropped = crop_or_pad_slice_to_size(label[:, :, zz], size, size)

            image_list_part2.append(image_n4[:, :, zz])
            label_list_part2.append(label[:, :, zz])

            write_buffer += 1

            if write_buffer >= MAX_WRITE_BUFFER:

                counter_to = counter_from + write_buffer

                _write_range_to_hdf5(data_part2,
                                    image_list_part2,
                                    label_list_part2,
                                    counter_from,
                                    counter_to)
                
                _release_tmp_memory(image_list_part2,
                                    label_list_part2)

                # update counters 
                counter_from = counter_to
                write_buffer = 0


    #logging.info('Writing remaining data')
    #counter_to = counter_from + write_buffer

    #_write_range_to_hdf5(data, image_list, label_list, counter_from, counter_to)
   # _release_tmp_memory(image_list, label_list)


   
    hdf5_file_part2.create_dataset('nx', data=np.asarray(nx_list_part2, dtype=np.uint16))
    hdf5_file_part2.create_dataset('ny', data=np.asarray(ny_list_part2, dtype=np.uint16))
    hdf5_file_part2.create_dataset('nz', data=np.asarray(nz_list_part2, dtype=np.uint16))
    hdf5_file_part2.create_dataset('px', data=np.asarray(px_list_part2, dtype=np.float32))
    hdf5_file_part2.create_dataset('py', data=np.asarray(py_list_part2, dtype=np.float32))
    hdf5_file_part2.create_dataset('pz', data=np.asarray(pz_list_part2, dtype=np.float32))
    hdf5_file_part2.create_dataset('patnames', data=np.asarray(pat_names_list_part2, dtype="S20"))
    
    # After test train loop:
    logging.info('Training part2 loop done')
    hdf5_file_part2.close()


    hdf5_file_part3 = h5py.File(train_file_path_part3, "w")
    data_part3 = {}
    data_part3['images'] = hdf5_file_part3.create_dataset("images", list((size,size)) + [num_slices_part3], dtype=np.float32)
    data_part3['labels'] = hdf5_file_part3.create_dataset("labels", list((size,size)) + [num_slices_part3], dtype=np.uint8)


    for folder in folder_list_part3:

        patient_counter += 1

        logging.info('================================')
        logging.info('Doing: %s' % folder)
        patname = folder.split('/')[-1]
        pat_names_list_part3.append(patname)


        image_t1, _, image_t1_hdr = utils.load_nii(folder + f'/{patname}_t1.nii.gz')
        image_t1ce, _, image_t1ce_hdr = utils.load_nii(folder + f'/{patname}_t1ce.nii.gz')
        image_t2, _, image_t2_hdr = utils.load_nii(folder + f'/{patname}_t2.nii.gz')
        image_flair, _, image_flair_hdr = utils.load_nii(folder + f'/{patname}_flair.nii.gz')


        px_list_part3.append(float(image_t1_hdr.get_zooms()[0]))
        py_list_part3.append(float(image_t1_hdr.get_zooms()[1]))
        pz_list_part3.append(float(image_t1_hdr.get_zooms()[2]))



        nifti_img_path = preprocessing_folder + '/Individual_NIFTI/' + patname + '/'
        if not os.path.exists(nifti_img_path):
            utils.makefolder(nifti_img_path)
        #utils.save_nii(img_path = nifti_img_path + '_img_t1.nii.gz', data = image_t1, affine = np.eye(4))
        #utils.save_nii(img_path = nifti_img_path + '_img_t1ce.nii.gz', data = image_t1ce, affine = np.eye(4))
        #utils.save_nii(img_path = nifti_img_path + '_img_t2.nii.gz', data = image_t2, affine = np.eye(4))
        #utils.save_nii(img_path = nifti_img_path + '_img_flair.nii.gz', data = image_flair, affine = np.eye(4))


        # ================================
        # do bias field correction
        # ================================
        input_img_t1 = nifti_img_path + patname + '_img_t1.nii.gz'
        output_img_t1 = nifti_img_path + patname + '_img_t1_n4.nii.gz'
        input_img_t1ce = nifti_img_path + patname + '_img_t1ce.nii.gz'
        output_img_t1ce = nifti_img_path + patname + '_img_t1ce_n4.nii.gz'
        input_img_t2 = nifti_img_path + patname + '_img_t2.nii.gz'
        output_img_t2 = nifti_img_path + patname + '_img_t2_n4.nii.gz'
        input_img_flair = nifti_img_path + patname + '_img_flair.nii.gz'
        output_img_flair = nifti_img_path + patname + '_img_flair_n4.nii.gz'


        # If bias corrected image does not exist, do it now
        for input_img, output_img in zip([input_img_t1, input_img_t1ce, input_img_t2, input_img_flair], [output_img_t1, output_img_t1ce, output_img_t2, output_img_flair]):
            if os.path.isfile(output_img):
                img = utils.load_nii(img_path = output_img)[0]
            else:
                subprocess.call(["/itet-stor/arismu/bmicdatasets_bmicnas01/Sharing/N4_th", input_img, output_img])
                img = utils.load_nii(img_path = output_img)[0]

            if input_img == input_img_t1:
                img_t1_n4 = img
                img_t1_n4 = utils.normalise_image(img_t1_n4, norm_type='div_by_max')
            elif input_img == input_img_t1ce:
                img_t1ce_n4 = img
                img_t1ce_n4 = utils.normalise_image(img_t1ce_n4, norm_type='div_by_max')
            elif input_img == input_img_t2:
                img_t2_n4 = img
                img_t2_n4 = utils.normalise_image(img_t2_n4, norm_type='div_by_max')
            elif input_img == input_img_flair:
                img_flair_n4 = img
                img_flair_n4 = utils.normalise_image(img_flair_n4, norm_type='div_by_max')

        image_n4 = np.concatenate((img_t1_n4, img_t1ce_n4), axis = 2)
        image_n4 = np.concatenate((image_n4, img_t2_n4), axis = 2)
        image_n4 = np.concatenate((image_n4, img_flair_n4), axis = 2)

        nx_list_part3.append(image_n4.shape[0])
        ny_list_part3.append(image_n4.shape[1])
        nz_list_part3.append(image_n4.shape[2])
            

        # ================================    
        # read the labels
        # ================================   
        
        
        label, _, _ = utils.load_nii(folder + f'/{patname}_seg.nii.gz')

        label_temp = np.concatenate((label, label), axis = 2) #concatenate 3 times since all 4 image types share the same label
        label_temp = np.concatenate((label_temp, label), axis = 2)
        label = np.concatenate((label_temp, label), axis = 2)
        
        if not os.path.isfile(nifti_img_path + patname + '_lbl.nii.gz'):
            utils.save_nii(img_path = nifti_img_path + patname + '_lbl.nii.gz', data = label, affine = np.eye(4))

        ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################

        for zz in range(image_n4.shape[2]):

            #no rescaling needed since all images (SD & TDs) have same scale/dimensions already

            #image_cropped = crop_or_pad_slice_to_size(img_normalised[:, :, zz], size, size)
            #label_cropped = crop_or_pad_slice_to_size(label[:, :, zz], size, size)

            image_list_part3.append(image_n4[:, :, zz])
            label_list_part3.append(label[:, :, zz])

            write_buffer += 1

            if write_buffer >= MAX_WRITE_BUFFER:

                counter_to = counter_from + write_buffer

                _write_range_to_hdf5(data_part3,
                                    image_list_part3,
                                    label_list_part3,
                                    counter_from,
                                    counter_to)
                
                _release_tmp_memory(image_list_part3,
                                    label_list_part3)

                # update counters 
                counter_from = counter_to
                write_buffer = 0


    #logging.info('Writing remaining data')
    #counter_to = counter_from + write_buffer

    #_write_range_to_hdf5(data, image_list, label_list, counter_from, counter_to)
   # _release_tmp_memory(image_list, label_list)


   
    hdf5_file_part3.create_dataset('nx', data=np.asarray(nx_list_part3, dtype=np.uint16))
    hdf5_file_part3.create_dataset('ny', data=np.asarray(ny_list_part3, dtype=np.uint16))
    hdf5_file_part3.create_dataset('nz', data=np.asarray(nz_list_part3, dtype=np.uint16))
    hdf5_file_part3.create_dataset('px', data=np.asarray(px_list_part3, dtype=np.float32))
    hdf5_file_part3.create_dataset('py', data=np.asarray(py_list_part3, dtype=np.float32))
    hdf5_file_part3.create_dataset('pz', data=np.asarray(pz_list_part3, dtype=np.float32))
    hdf5_file_part3.create_dataset('patnames', data=np.asarray(pat_names_list_part3, dtype="S20"))
    
    # After test train loop:
    logging.info('Training part3 loop done')
    hdf5_file_part3.close()




# ===============================================================
# Helper function to write a range of data to the hdf5 datasets
# ===============================================================
def _write_range_to_hdf5(hdf5_data,
                         img_list,
                         mask_list,
                         counter_from,
                         counter_to):

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))

    img_arr = np.asarray(img_list, dtype=np.float32)
    lab_arr = np.asarray(mask_list, dtype=np.uint8)

    img_arr = np.swapaxes(img_arr, 0, 2)
    lab_arr = np.swapaxes(lab_arr, 0, 2)

    hdf5_data['images'][..., counter_from : counter_to] = img_arr
    hdf5_data['labels'][..., counter_from : counter_to] = lab_arr

# ===============================================================
# Helper function to reset the tmp lists and free the memory
# ===============================================================
def _release_tmp_memory(img_list,
                        mask_list):

    img_list.clear()
    mask_list.clear()
    gc.collect()




def load_validation_data(input_folder,
              preproc_folder,
              size,
              target_resolution,
              force_overwrite = False):

    # ===============================
    # create the pre-processing folder, if it does not exist
    # ===============================
    utils.makefolder(preproc_folder)    
    
    # ===============================
    # file to create or directly read if it already exists
    # ===============================

    val_file_name = 'validation_data_NEW.hdf5' 
    val_file_path = os.path.join(preproc_folder, val_file_name)

    
    # ===============================
    # if the images have not already been extracted, do so
    # ===============================
    #if not os.path.exists(train_file_path_part1 or train_file_path_part2 or train_file_path_part3) or force_overwrite:
    if not os.path.exists(val_file_path) or force_overwrite:

        logging.info('This configuration of protocol and data indices has not yet been preprocessed')
        logging.info('Preprocessing now...')
        prepare_validation_data(input_folder,
                     preproc_folder,
                     val_file_path,
                     size,
                     target_resolution
                     )
    else:        
        logging.info('Already preprocessed this configuration. Loading now...')
        
    return h5py.File(val_file_path, 'r')



def prepare_validation_data(input_folder,
                 preprocessing_folder,
                 val_file_path, 
                 size,
                 target_resolution
                 ):

    # ===============================
    # create a hdf5 file
    # ===============================
    hdf5_file_val = h5py.File(val_file_path, "w")
    

    # ===============================
    # read all the patient folders from the base input folder
    # ===============================


    logging.info('Counting files and parsing meta data...')

    training_folder = input_folder + 'MICCAI_FeTS2021_TrainingData/'

    
    folder_list_val = []

    validation_ids = [54, 53, 51, 50, 49, 52, 47, 35, 36, 37, 38, 48, 40, 41, 39, 43, 44, 45, 46, 42]
    num_slices_val = 0


    for folder in os.listdir(training_folder):
        if not (folder.lower().endswith('.csv') or folder.lower().endswith('.md')):
            folder_path = os.path.join(training_folder, folder)  
            patient_id = int(folder.split('_')[-1])      
            if os.path.isdir(folder_path):
                
                if patient_id in validation_ids:
                    folder_list_val.append(folder_path)
                    for _, _, fileList in os.walk(folder_path):
                        for filename in fileList:
                                if filename.lower().endswith('t1.nii.gz'):
                                    image_t1, _, _ = utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices_val += image_t1.shape[2]
                                elif filename.lower().endswith('t1ce.nii.gz'):
                                    image_t1ce, _, _ = utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices_val += image_t1ce.shape[2]
                                elif filename.lower().endswith('t2.nii.gz'):
                                    image_t2, _, _= utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices_val += image_t2.shape[2]
                                elif filename.lower().endswith('flair.nii.gz'):
                                    image_flair, _, _ = utils.load_nii(training_folder + folder + '/' + filename)
                                    num_slices_val += image_flair.shape[2]

    # ===============================
    # Create datasets for images and labels
    # ===============================

   
    #num_slices = count_slices(folder_list, idx_start, idx_end)
    #data['images'] = hdf5_file.create_dataset("images", list((size,size)) + [num_slices], dtype=np.float32)
    #data['labels'] = hdf5_file.create_dataset("labels", list((size,size)) + [num_slices], dtype=np.uint8)
    
    # ===============================
    # initialize lists
    # ===============================    
    
    label_list_val = []
    image_list_val = []
    nx_list_val = []
    ny_list_val = []
    nz_list_val = []
    px_list_val = []
    py_list_val = []
    pz_list_val= []
    pat_names_list_val = []

    # ===============================        
    # ===============================        
    logging.info('Parsing image files')
    
        
    patient_counter = 0
    write_buffer = 0
    counter_from = 0

    
    data_val = {}
    data_val['images'] = hdf5_file_val.create_dataset("images", list((size,size)) + [num_slices_val], dtype=np.float32)
    data_val['labels'] = hdf5_file_val.create_dataset("labels", list((size,size)) + [num_slices_val], dtype=np.uint8)



    for folder in folder_list_val:

        patient_counter += 1

        logging.info('================================')
        logging.info('Doing: %s' % folder)
        patname = folder.split('/')[-1]
        pat_names_list_val.append(patname)


        image_t1, _, image_t1_hdr = utils.load_nii(folder + f'/{patname}_t1.nii.gz')
        image_t1ce, _, image_t1ce_hdr = utils.load_nii(folder + f'/{patname}_t1ce.nii.gz')
        image_t2, _, image_t2_hdr = utils.load_nii(folder + f'/{patname}_t2.nii.gz')
        image_flair, _, image_flair_hdr = utils.load_nii(folder + f'/{patname}_flair.nii.gz')


        px_list_val.append(float(image_t1_hdr.get_zooms()[0]))
        py_list_val.append(float(image_t1_hdr.get_zooms()[1]))
        pz_list_val.append(float(image_t1_hdr.get_zooms()[2]))



        nifti_img_path = preprocessing_folder + '/Individual_NIFTI/' + patname + '/'
        if not os.path.exists(nifti_img_path):
            utils.makefolder(nifti_img_path)
        #utils.save_nii(img_path = nifti_img_path + '_img_t1.nii.gz', data = image_t1, affine = np.eye(4))
        #utils.save_nii(img_path = nifti_img_path + '_img_t1ce.nii.gz', data = image_t1ce, affine = np.eye(4))
        #utils.save_nii(img_path = nifti_img_path + '_img_t2.nii.gz', data = image_t2, affine = np.eye(4))
        #utils.save_nii(img_path = nifti_img_path + '_img_flair.nii.gz', data = image_flair, affine = np.eye(4))


        # ================================
        # do bias field correction
        # ================================
        input_img_t1 = nifti_img_path + patname + '_img_t1.nii.gz'
        output_img_t1 = nifti_img_path + patname + '_img_t1_n4.nii.gz'
        input_img_t1ce = nifti_img_path + patname + '_img_t1ce.nii.gz'
        output_img_t1ce = nifti_img_path + patname + '_img_t1ce_n4.nii.gz'
        input_img_t2 = nifti_img_path + patname + '_img_t2.nii.gz'
        output_img_t2 = nifti_img_path + patname + '_img_t2_n4.nii.gz'
        input_img_flair = nifti_img_path + patname + '_img_flair.nii.gz'
        output_img_flair = nifti_img_path + patname + '_img_flair_n4.nii.gz'


        # If bias corrected image does not exist, do it now
        for input_img, output_img in zip([input_img_t1, input_img_t1ce, input_img_t2, input_img_flair], [output_img_t1, output_img_t1ce, output_img_t2, output_img_flair]):
            if os.path.isfile(output_img):
                img = utils.load_nii(img_path = output_img)[0]
            else:
                subprocess.call(["/itet-stor/arismu/bmicdatasets_bmicnas01/Sharing/N4_th", input_img, output_img])
                img = utils.load_nii(img_path = output_img)[0]

            if input_img == input_img_t1:
                img_t1_n4 = img
                img_t1_n4 = utils.normalise_image(img_t1_n4, norm_type='div_by_max')
            elif input_img == input_img_t1ce:
                img_t1ce_n4 = img
                img_t1ce_n4 = utils.normalise_image(img_t1ce_n4, norm_type='div_by_max')
            elif input_img == input_img_t2:
                img_t2_n4 = img
                img_t2_n4 = utils.normalise_image(img_t2_n4, norm_type='div_by_max')
            elif input_img == input_img_flair:
                img_flair_n4 = img
                img_flair_n4 = utils.normalise_image(img_flair_n4, norm_type='div_by_max')

        image_n4 = np.concatenate((img_t1_n4, img_t1ce_n4), axis = 2)
        image_n4 = np.concatenate((image_n4, img_t2_n4), axis = 2)
        image_n4 = np.concatenate((image_n4, img_flair_n4), axis = 2)

        nx_list_val.append(image_n4.shape[0])
        ny_list_val.append(image_n4.shape[1])
        nz_list_val.append(image_n4.shape[2])
            

        # ================================    
        # read the labels
        # ================================   
        
        
        label, _, _ = utils.load_nii(folder + f'/{patname}_seg.nii.gz')

        label_temp = np.concatenate((label, label), axis = 2) #concatenate 3 times since all 4 image types share the same label
        label_temp = np.concatenate((label_temp, label), axis = 2)
        label = np.concatenate((label_temp, label), axis = 2)
        
        if not os.path.isfile(nifti_img_path + patname + '_lbl.nii.gz'):
            utils.save_nii(img_path = nifti_img_path + patname + '_lbl.nii.gz', data = label, affine = np.eye(4))

        ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################

        for zz in range(image_n4.shape[2]):

            #no rescaling needed since all images (SD & TDs) have same scale/dimensions already

            #image_cropped = crop_or_pad_slice_to_size(img_normalised[:, :, zz], size, size)
            #label_cropped = crop_or_pad_slice_to_size(label[:, :, zz], size, size)

            image_list_val.append(image_n4[:, :, zz])
            label_list_val.append(label[:, :, zz])

            write_buffer += 1

            if write_buffer >= MAX_WRITE_BUFFER:

                counter_to = counter_from + write_buffer

                _write_range_to_hdf5(data_val,
                                    image_list_val,
                                    label_list_val,
                                    counter_from,
                                    counter_to)
                
                _release_tmp_memory(image_list_val,
                                    label_list_val)

                # update counters 
                counter_from = counter_to
                write_buffer = 0


    #logging.info('Writing remaining data')
    #counter_to = counter_from + write_buffer

    #_write_range_to_hdf5(data, image_list, label_list, counter_from, counter_to)
   # _release_tmp_memory(image_list, label_list)


   
    hdf5_file_val.create_dataset('nx', data=np.asarray(nx_list_val, dtype=np.uint16))
    hdf5_file_val.create_dataset('ny', data=np.asarray(ny_list_val, dtype=np.uint16))
    hdf5_file_val.create_dataset('nz', data=np.asarray(nz_list_val, dtype=np.uint16))
    hdf5_file_val.create_dataset('px', data=np.asarray(px_list_val, dtype=np.float32))
    hdf5_file_val.create_dataset('py', data=np.asarray(py_list_val, dtype=np.float32))
    hdf5_file_val.create_dataset('pz', data=np.asarray(pz_list_val, dtype=np.float32))
    hdf5_file_val.create_dataset('patnames', data=np.asarray(pat_names_list_val, dtype="S20"))
    
    # After test train loop:
    logging.info('Validation loop done')
    hdf5_file_val.close()