import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from mlxtend.evaluate import permutation_test

unet = [0.003949991, 
0.118263058, 
0.194805896, 
0.31015765, 
0.732680957]

unwt = [0.00484978,
0.17236218,
0.262858438,
0.38636308,
0.838941316]

significance_level = 0.05

def paired_two_sample_test():
    p_value = permutation_test(
    unet, unwt, paired=True, method="approximate", seed=0, num_rounds=100000
)
    significant = p_value < significance_level
    print(f'The p-value in the Paired two-sided randomization test is {p_value}')
    if significant:
        print(f'Paired two-sided randomization test: The difference is statistically significant!')
    else:
        print("Paired two-sided randomization test: The difference is not statistically significant!")


if __name__ == "__main__":

    paired_two_sample_test()