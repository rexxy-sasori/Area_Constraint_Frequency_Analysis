import glob
import os
import time
from datetime import datetime
from multiprocessing import Pool

import matplotlib.pyplot as plt
import torch
from sklearn import metrics
from tqdm import tqdm


def plot_roc_area(results, roc_name, auc_name):
    areas = []
    freqs = []
    ks = []
    plt.figure(figsize=(10, 5))
    for idx, result in enumerate(results):
        area = metrics.auc(result.roc.fpr, result.roc.tpr)
        areas.append(area)
        freqs.append(float(result.usr_configs.signal.freqs[0]))
        f0 = result.usr_configs.signal.freqs[0]
        k0 = result.usr_configs.signal.freqs[0] * result.usr_configs.signal.block_size / result.usr_configs.signal.fs
        ks.append(k0)
        if idx % 25 == 0:
            plt.plot(result.roc.fpr, result.roc.tpr, label='$k_o$: '+str(k0)+', '
                                                           ''
                                                           ' f_0: {}$'.format(k0, '{0:.5g} Hz'.format(f0)))

    plt.grid()
    plt.xlabel('$p_{fp}$', fontsize=15)
    plt.ylabel('$p_{tp}$', fontsize=15)
    plt.tick_params('both', labelsize=15)
    plt.legend(loc='lower right', fontsize=15)
    plt.savefig(roc_name)
    plt.clf()
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(ks, areas, marker='o', markersize=5)
    # plt.scatter(ks[::5], areas[::5],marker='*', s=600,c='r')
    plt.xlabel('$k_o$', fontsize=15)
    plt.ylabel('Area under ROC Curve', fontsize=15)
    plt.grid()
    plt.tick_params('both', labelsize=15)
    plt.savefig(auc_name)
    plt.clf()
    plt.close()


def plot_auc_roc_single_experiment(result_dir):
    if 'noise_level_0.1' in result_dir:
        return

    results_identifier = os.path.join(result_dir, '*tar')
    result_paths = glob.glob(results_identifier)

    if len(result_paths) != 101:
        return

    print('working on dir: ', result_dir)

    plot_dir = os.path.join('/home/hgeng4/pmsp/plots', '/'.join(result_dir.lstrip('../').split('/')[1::]))

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    roc_file_name = os.path.join(plot_dir, 'roc.png')
    auc_file_name = os.path.join(plot_dir, 'auc.png')

    results = []
    for p in result_paths:
        result = torch.load(p).get('result')
        results.append(result)

    results = sorted(results, key=lambda x: x.usr_configs.signal.freqs[0])
    plot_roc_area(results, roc_file_name, auc_file_name)


if __name__ == '__main__':
    result_dirs = glob.glob('/home/hgeng4/pmsp/results/*/*/*/*/*/*')
    # for result_dir in result_dirs:
    #     if 'noise_level_0' in result_dir:
    #         continue
    #     if 'noise_level_0.1' in result_dir:
    #         continue
    #
    #     print('plotting results in {}'.format(result_dir))
    #     plot_auc_roc_single_experiment(result_dir, '*tar')

    pool = Pool(processes=32)
    s = time.time()
    print('staring at', datetime.now())
    for _ in tqdm(pool.map(plot_auc_roc_single_experiment, result_dirs), total=len(result_dirs)):
        pass
    e = time.time()
    print('end at', datetime.now())
    print('total time in second:', e - s)
