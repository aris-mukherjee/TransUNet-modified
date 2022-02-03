import numpy as np
import utils


def load_img():
    image_t1ce, _, _ = utils.load_nii('/scratch_net/biwidl217_second/arismu/Data_MT/MICCAI_FeTS2021_TrainingData/FeTS21_Training_001/FeTS21_Training_001_seg.nii.gz')

    print(np.unique(image_t1ce))


if __name__ == "__main__":
    load_img()