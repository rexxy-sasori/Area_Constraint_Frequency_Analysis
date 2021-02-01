import glob
import os

import matplotlib.pyplot as plt
import torch
from sklearn import metrics


def plot_roc_area(results, anchor_results, roc_name, roc_anchor_name, auc_name):
    areas, ks = roc_plot(results, roc_name)
    areas_anchor, ks_anchor = roc_plot(anchor_results, roc_anchor_name)

    plt.figure(figsize=(10, 5))
    plt.plot(ks, areas, marker='o', label='J-DHT')
    plt.plot(ks_anchor, areas_anchor, marker='o', label='DFT')

    plt.xlabel('$k_o$', fontsize=15)
    plt.ylabel('Area under ROC Curve', fontsize=15)
    plt.grid()
    plt.legend(fontsize=15)
    plt.tick_params('both', labelsize=15)
    plt.savefig(auc_name)
    plt.clf()
    plt.close()


def roc_plot(results, roc_name):
    areas = []
    freqs = []
    ks = []
    plt.figure(figsize=(10, 5))
    for idx, result in enumerate(results):
        area = metrics.auc(result.roc.fpr, result.roc.tpr)
        areas.append(area)

        f0 = result.usr_configs.signal.freqs[0]
        k0 = result.usr_configs.signal.freqs[0] * result.usr_configs.signal.block_size / result.usr_configs.signal.fs
        freqs.append(float(f0))
        ks.append(k0)

        if idx % 5 == 0:
            plt.plot(result.roc.fpr, result.roc.tpr, label='k_0: {}; f_0: {}'.format(k0, '{0:.5g} Hz'.format(f0)))
    plt.grid()
    plt.xlabel('$p_{fp}$', fontsize=15)
    plt.ylabel('$p_{tp}$', fontsize=15)
    plt.tick_params('both', labelsize=15)
    plt.legend(loc='lower right', fontsize=15)
    plt.savefig(roc_name)
    plt.clf()
    plt.close()
    return areas, ks


def plot_auc_roc_single_experiment(result_dir, anchor_dir, identifier='*tar'):
    results_identifier = os.path.join(result_dir, identifier)
    result_paths = glob.glob(results_identifier)

    anchor_paths = glob.glob(os.path.join(anchor_dir, identifier))

    plot_dir = os.path.join('../plots/compare_detection', '/'.join(result_dir.lstrip('./').split('/')[1::]))

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    roc_file_name = os.path.join(plot_dir, 'roc.png')
    roc_anchor_name = os.path.join(plot_dir, 'equi_dft_roc.png')
    auc_file_name = os.path.join(plot_dir, 'auc.png')

    results = []
    for p in result_paths:
        result = torch.load(p).get('result')
        results.append(result)

    anchor_results = []
    for p in anchor_paths:
        result = torch.load(p).get('result')
        anchor_results.append(result)

    if len(results) != len(anchor_results):
        return

    results = sorted(results, key=lambda x: x.usr_configs.signal.freqs[0])
    anchor_results = sorted(anchor_results, key=lambda x: x.usr_configs.signal.freqs[0])
    plot_roc_area(results, anchor_results, roc_file_name, roc_anchor_name, auc_file_name)


if __name__ == '__main__':
    result_dirs = glob.glob('../results/*/*/*/*/*/*')

    constraint_func = lambda x: 'switched' in x or 'ml' in x
    result_dirs = list(filter(constraint_func, result_dirs))

    for result_dir in result_dirs:
        result_dir = result_dir.lstrip('../')
        result_dir_levels = result_dir.split('/')

        anchor_dir_levels = result_dir_levels
        anchor_dir_levels[1] = 'Fmethod_fft'

        anchor_dir_levels[5] = 'L_1'
        anchor_dir = '/'.join(anchor_dir_levels)

        print('plotting results in {}'.format(result_dir))
        plot_auc_roc_single_experiment(os.path.join('..', result_dir), os.path.join('..', anchor_dir), '*tar')
