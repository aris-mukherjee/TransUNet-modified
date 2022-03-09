import numpy as np
import utils


def load_img():

    for idx in ['001', '053', '006', '353', '201']:
        image_t1, _, image_t1_hdr = utils.load_nii(f'/scratch_net/biwidl217_second/arismu/Data_MT/MICCAI_FeTS2021_TrainingData/FeTS21_Training_{idx}/FeTS21_Training_{idx}_t1.nii.gz')
        


        print(f'INDEX: {idx}')
        print('--------------------')
        
        print(f'x: {image_t1_hdr.get_zooms()[0]}')
        print(f'y: {image_t1_hdr.get_zooms()[1]}')
        print(f'z: {image_t1_hdr.get_zooms()[2]}')


        onelist_elem = []
        onelist_lab = []

        for elem, lab in zip[foreground_list_arr, label_list]:
            if 0<elem<0.2:
                onelist_elem.append(elem)
                onelist_lab.append(lab)

        for elem, lab in zip[onelist_elem, onelist_lab]:
            if elem 

        


if __name__ == "__main__":
    load_img()