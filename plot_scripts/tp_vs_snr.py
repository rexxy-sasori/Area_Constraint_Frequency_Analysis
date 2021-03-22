import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d

from core import utils


def find_tp_rate(result, search_fpr=0.1):
    roc = result.roc
    roc_func = interp1d(roc.fpr, roc.tpr)
    return roc_func(search_fpr)


def tp_vs_snr(parent_dir, search_fpr=0.1, method='fft'):
    noise_level_dirs = glob.glob(os.path.join(parent_dir, '*'))
    noise_levels = []
    noise_level_tprs = {}
    tprs = []

    freqs = []
    compute_output_powers = []
    flag = 0

    for idx, noise_level_dir in enumerate(noise_level_dirs):
        noise_level_str = os.path.split(noise_level_dir)[1]
        noise_level_str = noise_level_str.lstrip('inde_noise_level_')

        if noise_level_str == '0' or noise_level_str == '10':
            continue

        print(noise_level_dir)
        noise_level = float(noise_level_str)
        noise_levels.append(noise_level)

        result_paths = glob.glob(os.path.join(noise_level_dir, '*tar'))
        results = sorted([torch.load(p).get('result') for p in result_paths],
                         key=lambda x: x.usr_configs.signal.freqs[0])

        if flag == 0:
            for r in results:
                freq = r.usr_configs.signal.freqs[0]
                fs = r.usr_configs.signal.fs
                phase = r.usr_configs.signal.phases[0]
                N = r.usr_configs.signal.block_size
                signal_power_kernel = utils.__SIGNAL_POWER__.get(method)
                power = signal_power_kernel(freq, phase, fs, N)
                freqs.append(freq)
                compute_output_powers.append(power)
                flag = 1

        noise_level_tprs[noise_level] = []
        for r in results:
            tpr = find_tp_rate(r, search_fpr)
            noise_level_tprs[noise_level].append(tpr)

    noise_levels = sorted(noise_levels)
    for noise_level in noise_levels:
        tprs.append(np.array(noise_level_tprs[noise_level]))

    return np.array(noise_levels), np.array(tprs), np.array(freqs), np.array(compute_output_powers)


def plot_tpr_vs_noise_level(noise_levels, tprs, freqs, compute_output_power, how_often=25, fpr_subj=0.05, N=16,
                            fs=2000, L=1):
    num_freqs = tprs.shape[1]
    print(num_freqs)
    plt.figure(figsize=(10, 5))
    k0s = np.linspace(3, 4, num_freqs)
    for idx, f in enumerate(range(num_freqs)):
        if idx % how_often == 0:
            plt.plot(10 * np.log10(compute_output_power[idx]) - 10 * np.log10(noise_levels / N / L), tprs[:, idx],
                     marker='o', markersize=5,
                     label='$k_o$ ' + str(k0s[idx]))

    plt.grid()
    plt.xlabel('Detector Input SNR(dB)', fontsize=15)
    plt.ylabel('$p_{tp}$', fontsize=15)
    plt.title('fpr constraint: {}'.format(fpr_subj), fontsize=15)
    plt.tick_params('both', labelsize=15)
    plt.legend(loc='lower right', fontsize=15)
    plt.ylim([-0.1, 1.1])
    plt.savefig('/home/hgeng4/pmsp/plots/tpr_snr.png')
    plt.clf()
    plt.close()

    plt.figure(figsize=(10, 5))
    k0s = np.linspace(3, 4, num_freqs)
    for idx, f in enumerate(range(num_freqs)):
        if idx % how_often == 0:
            input_snr = - 10 * np.log10(noise_levels)
            output_snr = 10 * np.log10(compute_output_power[idx]) - 10 * np.log10(noise_levels / N / L)
            plt.plot(input_snr, output_snr,
                     marker='o', markersize=5,
                     label='$k_o$ ' + str(k0s[idx]))
    plt.grid()
    plt.xlabel('Input SNR(dB)', fontsize=15)
    plt.ylabel('Detector Input SNR(dB)', fontsize=15)
    plt.title('Detector Input SNR (dB) vs. Input SNR (dB)', fontsize=15)
    plt.tick_params('both', labelsize=15)
    plt.legend(loc='lower right', fontsize=15)
    plt.savefig('/home/hgeng4/pmsp/plots/dnr_snr.png')
    plt.clf()
    plt.close()


def dht_vs_dft_different_snr(dft_data, dht_data):
    num_freqs = dft_plot_data.tprs.shape[1]
    print(num_freqs)
    plt.figure(figsize=(10, 5))
    k0s = np.linspace(3, 4, num_freqs)

    num_plot = 0
    for idx, f in enumerate(range(num_freqs)):
        if idx % dft_data.how_often == 0 and num_plot < 2:
            plt.plot(-10 * np.log10(dft_data.noise_levels), dft_data.tprs[:, idx],
                     marker='o', markersize=5,
                     label='$k_o$ ' + str(k0s[idx]) + ' method: ' + dft_data.method)
            num_plot += 1

    num_plot = 0
    for idx, f in enumerate(range(num_freqs)):
        if idx % dht_data.how_often == 0 and num_plot < 2:
            plt.plot(-10 * np.log10(dht_data.noise_levels), dht_data.tprs[:, idx],
                     marker='o', markersize=5,
                     label='$k_o$ ' + str(k0s[idx]) + ' method: ' + dht_data.method)
            num_plot += 1

    plt.grid()
    plt.xlabel('Input SNR(dB)', fontsize=15)
    plt.ylabel('$p_{tp}$', fontsize=15)

    plt.tick_params('both', labelsize=15)
    plt.legend(loc='upper left', fontsize=15)
    plt.savefig('/home/hgeng4/pmsp/plots/tpr_snr.png')
    plt.clf()
    plt.close()


class PlotData:
    def __init__(self, noise_levels, tprs, freqs,
                 compute_output_power, how_often=25,
                 fpr_subj=0.05, N=16, fs=2000, L=1, method='DFT'):
        self.noise_levels = noise_levels
        self.tprs = tprs
        self.freqs = freqs
        self.compute_out_power = compute_output_power
        self.how_often = how_often
        self.fpr_subj = fpr_subj
        self.N = N
        self.fs = fs
        self.L = L
        self.method = method


if __name__ == '__main__':
    fpr_subj = 0.05
    fs = 2000
    N = 16
    L = 10
    fft_dirname = '/home/hgeng4/pmsp/results/Fmethod_fft/detection_ml/phi_0.7853981633974483/N_16/L_' + str(L)
    fht_dirname = '/home/hgeng4/pmsp/results/Fmethod_fht_jitter/detection_ml/phi_0.7853981633974483/N_16/L_' + str(L)
    noise_levels_fft, tprs_fft, freqs_fft, out_power_fft = tp_vs_snr(fft_dirname, fpr_subj, 'fft')
    noise_levels_fht, tprs_fht, freqs_fht, out_power_fht = tp_vs_snr(fht_dirname, fpr_subj, 'fht_jitter')

    dft_plot_data = PlotData(
        noise_levels=noise_levels_fft, tprs=tprs_fft, freqs=freqs_fft,
        compute_output_power=out_power_fft, how_often=25,
        fpr_subj=0.05, N=16, fs=2000, L=1, method='DFT'
    )

    dht_plot_data = PlotData(
        noise_levels=noise_levels_fht, tprs=tprs_fht, freqs=freqs_fht,
        compute_output_power=out_power_fht, how_often=25,
        fpr_subj=0.05, N=16, fs=2000, L=1, method='J-DHT'
    )

    dht_vs_dft_different_snr(dft_plot_data, dht_plot_data)