import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.image as mpimg


def create_subplots():

    img = mpimg.imread('/scratch_net/biwidl217_second/arismu/Data_MT/img_BIDMC_case5_slice25.png')
    pred_soft = mpimg.imread('/scratch_net/biwidl217_second/arismu/Data_MT/pred_soft_BIDMC_REVISED_ADAM_case5_slice25.png')
    pred_hard = mpimg.imread('/scratch_net/biwidl217_second/arismu/Data_MT/BIDMC_REVISED_TU_ADAM_hard_pred_case5_slice25_seed100.png')
    gt = mpimg.imread('/scratch_net/biwidl217_second/arismu/Data_MT/gt_BIDMC_case5_slice25.png')

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("TU like UNET on BIDMC Dataset, case 5, slice 25", fontsize=16)
    axs[0, 0].imshow(img)
    axs[0, 0].set_title("Image")
    axs[0, 0].axis('off')
    axs[0, 1].imshow(gt)
    axs[0, 1].set_title("Ground Truth")
    axs[0, 1].axis('off')
    axs[1, 0].imshow(pred_soft)
    axs[1, 0].set_title("Soft prediction")
    axs[1, 0].axis('off')
    axs[1, 1].imshow(pred_hard)
    axs[1, 1].set_title("Hard prediction (prob > 0.5)")
    axs[1, 1].axis('off')

    fig.savefig('/scratch_net/biwidl217_second/arismu/Data_MT/SUBPLOT_BIDMC.png')


if __name__ == "__main__":

    create_subplots()