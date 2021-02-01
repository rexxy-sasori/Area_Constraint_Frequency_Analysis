import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import core


def get_estimates(paths):
    estimates_table = []
    results = []
    for p in paths:
        result = torch.load(p).get('result')
        results.append(result)

    results = sorted(results, key=lambda x: x.usr_configs.signal.freqs[0])

    for result in results:
        if type(result.scores) == core.detector.MLEstimates:
            nd = len(result.scores.scores)
            positive_scores = result.scores.scores[0:int(nd / 2):]
        else:
            nd = len(result.scores)
            positive_scores = result.scores[0:int(nd / 2):]

        estimates_table.append(positive_scores)

    estimates_table = np.vstack(estimates_table)
    return estimates_table


def get_error(scores):
    error = (2 * np.sqrt(scores) - 1)
    error_square = error ** 2
    mse = (error_square.mean(1))
    return mse


def plot_error(result_dir, anchor_dir, identifier='*tar'):
    result_paths = glob.glob(os.path.join(result_dir, identifier))
    anchor_paths = glob.glob(os.path.join(anchor_dir, identifier))

    plot_dir = os.path.join('../plots/compare_estimation', '/'.join(result_dir.lstrip('./').split('/')[1::]))
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    result_scores = get_estimates(result_paths)
    anchor_scores = get_estimates(anchor_paths)

    result_errors = get_error(result_scores)
    anchor_errors = get_error(anchor_scores)

    plt.figure(figsize=(10, 5))
    plt.plot(np.linspace(3, 4, len(result_errors)), result_errors, marker='o', markersize=5, label='Jittered DHT')
    plt.plot(np.linspace(3, 4, len(anchor_errors)), anchor_errors, marker='o', markersize=5, label='DFT (L=5)')
    plt.xlabel('$k_o$', fontsize=15)
    plt.ylabel('MSE Error', fontsize=15)
    plt.grid()
    plt.legend(fontsize=15)
    plt.tick_params('both', labelsize=15)
    plt.savefig(os.path.join(plot_dir, 'mse_compare.png'))
    plt.clf()
    plt.close()


if __name__ == '__main__':
    result_dirs = glob.glob('../results/*/*/*/*/*')

    constraint_func = lambda x: ('jitter' in x) \
                                and ('phi_0.7853981633974483' in x or 'phi_0' in x)
    result_dirs = list(filter(constraint_func, result_dirs))

    for result_dir in result_dirs:
        result_dir = result_dir.lstrip('../')
        result_dir_levels = result_dir.split('/')

        anchor_dir_levels = result_dir_levels
        anchor_dir_levels[1] = 'Fmethod_fft'
        anchor_dir_levels[4] = 'L_5'

        anchor_dir = '/'.join(anchor_dir_levels)

        print('plotting results in {}'.format(result_dir))
        plot_error(os.path.join('..', result_dir), os.path.join('..', anchor_dir), '*tar')
