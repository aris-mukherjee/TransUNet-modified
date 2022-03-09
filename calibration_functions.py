from scipy.integrate import simps
from numpy import trapz
import numpy as np
import matplotlib.pyplot as plt

def find_bin_values(pred_list, label_list):

    first_bin = []
    second_bin = []
    third_bin = []
    fourth_bin = []
    fifth_bin = []

    pred_list = list(pred_list)
    label_list = list(label_list)

    num_ones_first_bin = 0
    num_ones_second_bin = 0
    num_ones_third_bin = 0
    num_ones_fourth_bin = 0
    num_ones_fifth_bin = 0

    for elem, lab in list(zip(pred_list, label_list)):
        if 0 < elem <= 0.2:
            first_bin.append(lab)
            if lab == 1:
                num_ones_first_bin += 1
        elif 0.2 < elem <= 0.4:
            second_bin.append(lab)
            if lab == 1:
                num_ones_second_bin += 1
        elif 0.4 < elem <= 0.6:
            third_bin.append(lab)
            if lab == 1:
                num_ones_third_bin += 1  
        elif 0.6 < elem <= 0.8:
            fourth_bin.append(lab)
            if lab == 1:
                num_ones_fourth_bin += 1 
        elif 0.8 < elem <= 1:
            fifth_bin.append(lab)
            if lab == 1:
                num_ones_fifth_bin += 1 
        


    first_bin_frac_pos = float(num_ones_first_bin/len(first_bin))
    second_bin_frac_pos = float(num_ones_second_bin/len(second_bin))
    third_bin_frac_pos = float(num_ones_third_bin/len(third_bin))
    fourth_bin_frac_pos = float(num_ones_fourth_bin/len(fourth_bin))
    fifth_bin_frac_pos = float(num_ones_fifth_bin/len(fifth_bin))
        


    print("Fraction of positives:")
    print(f"First bin: {first_bin_frac_pos}")
    print(f"Second bin: {second_bin_frac_pos}")
    print(f"Third bin: {third_bin_frac_pos}")
    print(f"Fourth bin: {fourth_bin_frac_pos}")
    print(f"Fifth bin: {fifth_bin_frac_pos}")
    

    return first_bin_frac_pos, second_bin_frac_pos, third_bin_frac_pos, fourth_bin_frac_pos, fifth_bin_frac_pos 


def find_area(first_bin_frac_pos, second_bin_frac_pos, third_bin_frac_pos, fourth_bin_frac_pos, fifth_bin_frac_pos):

    y = [first_bin_frac_pos, second_bin_frac_pos, third_bin_frac_pos, fourth_bin_frac_pos, fifth_bin_frac_pos]

    # Compute the area using the composite trapezoidal rule.
    area = trapz(y, [0, 0.3, 0.5, 0.7, 1])
    print("Trapezoidal area =", area)
    print(f"Absolute difference: {abs(0.5-area)}")

    # Compute the area using the composite Simpson's rule.
    area = simps(y, [0, 0.3, 0.5, 0.7, 1])
    print("Simpson area =", area)
    print(f"Absolute difference: {abs(0.5-area)}")


def plot_roc_curve(fpr, tpr, roc_auc, dataset):
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(f'/scratch_net/biwidl217_second/arismu/Data_MT/plots/{dataset}.png')