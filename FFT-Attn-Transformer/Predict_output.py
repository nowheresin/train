import numpy as np
import torch.nn as nn

# column
def predict(output, dec_output):

    good_list = []
    bad_list = []
    disease_list = []

    softmax_func = nn.Softmax(dim=1)
    soft_output = softmax_func(output)

    output_cpu = soft_output.to('cpu')
    output_array = output_cpu.detach().numpy()

    list = []
    # print(output_array)
    for i in range(len(output_array)):

        max_index = 1*output_array[i][0] + 2*output_array[i][1] + 3*output_array[i][2] + 4*output_array[i][3] + 5*output_array[i][4] + 6*output_array[i][5] - 1

        list.append(max_index)

    column_size = 5
    output_two_dimensional_array = np.array(list).reshape(-1, column_size)
    output_two_dimensional_list = output_two_dimensional_array.tolist()



    dec_output_cpu = dec_output.to('cpu')
    dec_output_array = dec_output_cpu.detach().numpy()
    for dec_output_row in range(len(dec_output_array)):
        if np.array_equal(dec_output_array[dec_output_row], [3, 3, 3, 0, 2]):
            good_list.append(output_two_dimensional_list[dec_output_row])
        elif np.array_equal(dec_output_array[dec_output_row], [4, 4, 4, 0, 2]):
            bad_list.append(output_two_dimensional_list[dec_output_row])
        elif np.array_equal(dec_output_array[dec_output_row], [5, 5, 5, 0, 2]):
            disease_list.append(output_two_dimensional_list[dec_output_row])

    return good_list, bad_list, disease_list

def mean_column(numpy_2d):
    column_means = np.mean(numpy_2d, axis=0)
    return column_means

def var_means(lists, mean_list):
    result = [[(a - b)**2 for a, b in zip(row, mean_list)] for row in lists]
    var_list = mean_column(result)
    return np.mean(var_list)

def bias_means(mean_list, true_list):
    bias = [(a - b)**2 for a, b in zip(mean_list, true_list)]
    return sum(bias)

def generalization_error(bias_2, var):
    return bias_2 + var

def prdicted_label(lists):

    label_1 = [3, 3, 3, 3, 2]
    label_2 = [4, 4, 4, 4, 2]
    label_3 = [5, 5, 5, 5, 2]

    result_1 = [[(a - b) ** 2 for a, b in zip(row, label_1)] for row in lists]
    result_2 = [[(a - b) ** 2 for a, b in zip(row, label_2)] for row in lists]
    result_3 = [[(a - b) ** 2 for a, b in zip(row, label_3)] for row in lists]

    mean_row_1 = np.mean(result_1, axis=1)
    mean_row_2 = np.mean(result_2, axis=1)
    mean_row_3 = np.mean(result_3, axis=1)

    counts_1 = 0
    counts_2 = 0
    counts_3 = 0

    for count in range(len(mean_row_1)):
        if mean_row_1[count] < 0.5:
            counts_1 += 1
        elif mean_row_2[count] < 0.5:
            counts_2 += 1
        elif mean_row_3[count] < 0.5:
            counts_3 += 1
    return counts_1, counts_2, counts_3