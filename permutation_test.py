import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from mlxtend.evaluate import permutation_test

significance_level = 0.05
unwt = [0.458518, 0.777538, 0.67163, 0.69898, 0.616587, 0.048897, 0.642826, 0.807522, 0.72445, 0.613052, 0.463054, 0.772753, 0.72389, 0.65163, 0.727938, 0.327396, 0.566908, 0.725945, 0.717264, 0.580691, 0.613791, 0.825085, 0.797424, 0.725477, 0.74827, 0.695866, 0.763386, 0.713137, 0.695608, 0.479585]
treatment_no_dropout = [0.499914, 0.795847, 0.7514, 0.591276, 0.777144, 0.493248, 0.724345, 0.45155, 0.648023, 0.594509, 0.36199, 0.735969, 0.530033, 0.478607, 0.506635, 0.41718, 0.429825, 0.502253, 0.473225, 0.313795, 0.309765, 0.781212, 0.522434, 0.465672, 0.532981, 0.473109, 0.610229, 0.594966, 0.511591, 0.36174]
treatment_no_qkv = [0.494659, 0.79163, 0.666945, 0.67319, 0.729938, 0.485557, 0.787188, 0.539936, 0.541545, 0.495934, 0.677442, 0.831039, 0.80883, 0.720074, 0.761836, 0.604124, 0.737568, 0.641775, 0.820704, 0.752981, 0.574703, 0.816193, 0.764458, 0.833778, 0.780071, 0.375301, 0.805663, 0.582071, 0.649159, 0.66981]
treatment_1head64 = [0.459019, 0.828603, 0.766566, 0.559579, 0.688218, 0.376115, 0.753273, 0.582943, 0.563341, 0.668142, 0.575543, 0.802309, 0.802848, 0.681425, 0.723556, 0.557854, 0.670377, 0.690357, 0.735596, 0.57472, 0.697377, 0.844664, 0.873639, 0.668727, 0.800916, 0.510439, 0.800251, 0.663986, 0.852978, 0.407438]
treatment_no_attention = [0.698505, 0.812897, 0.741717, 0.788756, 0.779539, 0.527164, 0.799084, 0.65729, 0.797317, 0.775271, 0.586952, 0.818419, 0.738209, 0.764481, 0.787924, 0.249629, 0.816502, 0.576586, 0.791734, 0.669139, 0.338961, 0.77303, 0.711587, 0.609477, 0.711637, 0.427061, 0.624427, 0.596153, 0.534403, 0.474959]
unet = [0.463528, 0.79273, 0.749587, 0.694097, 0.748277, 0.260549, 0.469069, 0.762904, 0.633708, 0.454492, 0.574951, 0.805651, 0.725617, 0.681368, 0.802516, 0.48918, 0.663131, 0.719921, 0.702498, 0.605471, 0.45123, 0.678076, 0.634238, 0.819282, 0.740465, 0.46538, 0.54691, 0.810989, 0.685686, 0.48909]
unet_with_dropout = [0.223252, 0.794586, 0.529038, 0.632787, 0.64384, 0.220899, 0.506, 0.67398, 0.492639, 0.283389, 0.216028, 0.740836, 0.52928, 0.587262, 0.688057, 0.123113, 0.444432, 0.802208, 0.480324, 0.403814, 0.505493, 0.737003, 0.725516, 0.759577, 0.793436, 0.353538, 0.587981, 0.807728, 0.684061, 0.721871]


orig_result = statistics.mean(unet_with_dropout) - statistics.mean(unet)

all_data = unet + unet_with_dropout  

def perm_test(data):

    shuffled_data = np.random.permutation(data)
    new_control_group_avg = statistics.mean(shuffled_data[:len(shuffled_data)//2])
    new_treatment_group_avg = statistics.mean(shuffled_data[len(shuffled_data)//2:])

    return  new_treatment_group_avg - new_control_group_avg 

def plot_test_statistic(simulated_results, p_value):

    density_plot = sns.kdeplot(simulated_results, shade=True)
    density_plot.set(
    xlabel='Difference in Dice Score average (treatment - control)',
    ylabel='Proportion of Simulations',
    title='Unet with Dropout layers - Test Statistic Distribution '
)

    density_plot.axvline(
        x=orig_result, 
        color='red', 
        linestyle='--'
    )

    plt.text(-0.17, 1.8, f'p-value: {round(p_value, 3)}', size = 'small', color = 'black')

    plt.legend(
    labels=[f'Observed Difference', 'Simulated'], 
    loc='upper right'
)

    plt.show()
    plt.savefig('/scratch_net/biwidl217_second/arismu/Data_MT/plots/stat_test_unet_with_dropout.png')

def paired_two_sample_test():
    p_value = permutation_test(
    unet, unet_with_dropout, paired=True, method="approximate", seed=0, num_rounds=100000
)
    significant = p_value < significance_level
    print(f'The p-value in the Paired two-sided randomization test is {p_value}')
    if significant:
        print(f'Paired two-sided randomization test: The difference is statistically significant!')
    else:
        print("Paired two-sided randomization test: The difference is not statistically significant!")


if __name__ == "__main__":

    num_simulations = 100000
    simulated_results = []
    simulations_greater_than_observed = 0
    
    for _ in range(num_simulations):
        simulated_results.append(perm_test(all_data))

    for x in simulated_results:
        if x <= orig_result:
            simulations_greater_than_observed = simulations_greater_than_observed + 1

    num_simulations = len(simulated_results)
    p_value = simulations_greater_than_observed / num_simulations

    plot_test_statistic(simulated_results, p_value)

    significant = p_value < significance_level

    print(f'The p-value in the Two-sided randomization test is {p_value}')

    if significant:
        print(f'Two-sided randomization test: The difference is statistically significant!')
    else:
        print("Two-sided randomization test: The difference is not statistically significant!")

    paired_two_sample_test()

    