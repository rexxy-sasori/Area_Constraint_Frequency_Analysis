import glob
import os

import numpy as np
import torch
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

def find_tp_rate(result, search_fpr=0.1):
    roc = result.roc
    roc_func = interp1d(roc.fpr,roc.tpr)
    return roc_func(search_fpr)


def tp_vs_snr(parent_dir, search_fpr=0.1):
    noise_level_dirs = glob.glob(os.path.join(parent_dir, '*'))

    noise_levels = []
    noise_level_tprs = {}

    tprs = []

    for noise_level_dir in noise_level_dirs:
        noise_level_str = os.path.split(noise_level_dir)[1]
        noise_level_str = noise_level_str.lstrip('inde_noise_level_')

        if noise_level_str == '0' or noise_level_str == '10':
            continue
        print(noise_level_dir)
        noise_level = float(noise_level_str)
        noise_levels.append(noise_level)

        result_paths = glob.glob(os.path.join(noise_level_dir, '*tar'))
        results = sorted([torch.load(p).get('result') for p in result_paths], key=lambda x: x.usr_configs.signal.freqs[0])

        noise_level_tprs[noise_level] = []
        for r in results:
            tpr = find_tp_rate(r, search_fpr)
            noise_level_tprs[noise_level].append(tpr)

    noise_levels = sorted(noise_levels)
    for noise_level in noise_levels:
        tprs.append(np.array(noise_level_tprs[noise_level]))

    return np.array(noise_levels), np.array(tprs)


def plot_tpr_vs_noise_level(noise_levels, tprs, how_often=25, fpr_subj=0.05, N=16,fs=2000):
    num_freqs = tprs.shape[1]
    print(num_freqs)
    plt.figure(figsize=(10,5))

    k0s = np.linspace(3,4,num_freqs)
    for idx, f in enumerate(range(num_freqs)):
        if idx % how_often == 0:
            plt.plot(-10*np.log10(noise_levels), tprs[:, idx],marker='o',markersize=5, label = '$k_o$ '+str(k0s[idx]))

    plt.grid()
    plt.xlabel('Input SNR(dB)', fontsize=15)
    plt.ylabel('$p_{tp}$', fontsize=15)
    plt.title('fpr constraint: {}'.format(fpr_subj))
    plt.tick_params('both', labelsize=15)
    plt.legend(loc='lower right', fontsize=15)
    plt.savefig('/home/hgeng4/pmsp/plots/tpr_snr.png')
    plt.clf()
    plt.close()


if __name__ == '__main__':
    dirname = '/home/hgeng4/pmsp/results/Fmethod_fft/detection_ml/phi_0/N_16/L_1'
    fpr_subj = 0.05
    fs=2000
    N=16
    noise_levels, tprs = tp_vs_snr(dirname, search_fpr=fpr_subj)
    plot_tpr_vs_noise_level(noise_levels,tprs, fpr_subj=fpr_subj, N=N,fs=fs)




















