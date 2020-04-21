import numpy as np
import torch


def test_irregular_sparsity(model):

    total_zeros = 0
    total_nonzeros = 0

    for name, weight in model.named_parameters():

        print(name)
        
        if 'conv' in name or 'feature' in name:
            zeros = np.sum(weight.cpu().detach().numpy() == 0)
            total_zeros += zeros
            non_zeros = np.sum(weight.cpu().detach().numpy() != 0)
            total_nonzeros += non_zeros
            zeros = np.sum(weight.cpu().detach().numpy() == 0)
            non_zero = np.sum(weight.cpu().detach().numpy() != 0)
            print("{}:irregular zeros: {}, irregular sparsity is: {:.4f}".format(
                name, zeros, zeros / (zeros + non_zero)))

    # print("---------------------------------------------------------------------------")
    # print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
    #     total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
    print("only consider conv layers, compression rate is: {:.4f}".format(
        (total_zeros+total_nonzeros) / total_nonzeros))
    print("===========================================================================\n\n")

    return (total_zeros+total_nonzeros) / total_nonzeros


def count_pattern_distribution(arr):
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v

        print(result)
    return result
