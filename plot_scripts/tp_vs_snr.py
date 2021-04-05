import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

from core import signal_generator

np.seterr(divide='raise')
color = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf'
]


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
                L = r.usr_configs.signal.num_blocks_avg
                output_power = signal_generator.get_output_signal_power(freq, phase, method, fs, N, L)
                freqs.append(freq)
                compute_output_powers.append(output_power)
                flag = 1

        noise_level_tprs[noise_level] = []
        for r in results:
            tpr = find_tp_rate(r, search_fpr)
            noise_level_tprs[noise_level].append(tpr)

    noise_levels = sorted(noise_levels)
    for noise_level in noise_levels:
        tprs.append(np.array(noise_level_tprs[noise_level]))

    return np.array(noise_levels), np.array(tprs), np.array(freqs), np.array(compute_output_powers)


def loop_through_plot_data_tpr(datas, num_freqs, k0s, freq_compare=3, marker='*'):
    for lidx, l in enumerate([1, 2, 5, 10]):
        data = datas[lidx]
        for idx, f in enumerate(range(num_freqs)):
            if k0s[idx] == freq_compare:
                plt.plot(-10 * np.log10(data.noise_levels), data.tprs[:, idx],
                         marker=marker, markersize=12, color=color[lidx],
                         label='$L=$' + str(l) + ', ' + data.method)


def compare_tpr(dft_datas, dht_datas, compare_k0 =3):
    num_freqs = dft_datas[0].tprs.shape[1]
    print(num_freqs)
    plt.figure(figsize=(10, 7))
    k0s = np.linspace(3, 4, num_freqs)

    compare_k0 = compare_k0
    loop_through_plot_data_tpr(dft_datas, num_freqs, k0s, compare_k0, 'o')
    loop_through_plot_data_tpr(dht_datas, num_freqs, k0s, compare_k0, 's')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=dft_datas[0].method, markerfacecolor='k', markersize=8),
        Line2D([0], [0], marker='s', color='w', label=dht_datas[0].method, markerfacecolor='k', markersize=8),

        Line2D([0], [0], color=color[0], lw=4, label='L=1'),
        Line2D([0], [0], color=color[1], lw=4, label='L=2'),
        Line2D([0], [0], color=color[2], lw=4, label='L=5'),
        Line2D([0], [0], color=color[3], lw=4, label='L=10'),
    ]

    plt.grid()
    plt.xlabel('Input SNR (dB)', fontsize=20)
    plt.ylabel('$p_{tp}$', fontsize=20)
    plt.xlim([-10,10])
    plt.tick_params('both', labelsize=20)
    plt.legend(handles=legend_elements, loc='best', fontsize=15, ncol=1)
    plt.savefig('/home/hgeng4/pmsp/plots/tpr_snr.png')
    plt.clf()
    plt.close()


def loop_through_plot_data_snrF(datas, num_freqs, k0s, freq_compare=3, marker='*', noise_power=None):
    for lidx, l in enumerate([1, 2, 5, 10]):
        data = datas[lidx]
        for idx, f in enumerate(range(num_freqs)):
            if k0s[idx] == freq_compare:
                try:
                    input_snr = - 10 * np.log10(data.noise_levels)

                    output_signal_power = data.compute_output_power
                    output_noise_power = noise_power[lidx, idx, :]
                    output_snr = 10*np.log10(output_signal_power[idx]/output_noise_power)
                except FloatingPointError:
                    continue

                plt.plot(
                    input_snr,
                    output_snr,
                    marker=marker,
                    markersize=8,
                    color=color[lidx],
                    label='$L=$' + str(l) + ', ' + data.method
                )


def compare_snrF(dft_datas, dht_datas, compare_k0, fft_noise_power, fht_noise_power, name1, name2):
    num_freqs = dft_datas[0].tprs.shape[1]
    print(num_freqs)
    plt.figure(figsize=(10, 5))
    k0s = np.linspace(3, 4, num_freqs)

    compare_k0 = compare_k0
    loop_through_plot_data_snrF(dft_datas, num_freqs, k0s, compare_k0, 'o', fft_noise_power)
    if compare_k0 != 3:
        loop_through_plot_data_snrF(dht_datas, num_freqs, k0s, compare_k0, '*', fht_noise_power)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=name1, markerfacecolor='k', markersize=8),
        Line2D([0], [0], marker='*', color='w', label=name2, markerfacecolor='k', markersize=15),

        Line2D([0], [0], color=color[0], lw=4, label='L=1'),
        Line2D([0], [0], color=color[1], lw=4, label='L=2'),
        Line2D([0], [0], color=color[2], lw=4, label='L=5'),
        Line2D([0], [0], color=color[3], lw=4, label='L=10'),
    ]

    plt.grid()
    plt.xlabel('$SNR_T$' + '(dB)', fontsize=15)
    plt.ylabel('$SNR_F$' + '(dB)', fontsize=15)

    plt.tick_params('both', labelsize=15)
    plt.legend(handles=legend_elements, loc='best', fontsize=15, ncol=1)
    plt.savefig('/home/hgeng4/pmsp/plots/snr_f_snr_t.png')
    plt.clf()
    plt.close()


def get_all_output_noise_power(Ls, noise_levels, freq_os, N=16, kernel='fft'):
    ret = np.zeros((len(Ls), len(freq_os), len(noise_levels)))

    for lidx, l in enumerate(Ls):
        for fidx, f0s in enumerate(freq_os):
            for nidx, noise_level in enumerate(noise_levels):
                print(l, f0s, noise_level, kernel)
                ret[lidx, fidx, nidx] = signal_generator.get_output_noise_power(
                    f0s, noise_level=noise_level, kernel=kernel, fs=2000, N=N, L=l
                )

    return ret



class PlotData:
    def __init__(self, noise_levels, tprs, freqs,
                 compute_output_power, how_often=25,
                 fpr_subj=0.05, N=16, fs=2000, L=1, method='DFT'):
        self.noise_levels = noise_levels
        self.tprs = tprs
        self.freqs = freqs
        self.compute_output_power = compute_output_power
        self.how_often = how_often
        self.fpr_subj = fpr_subj
        self.N = N
        self.fs = fs
        self.L = L
        self.method = method


if __name__ == '__main__':
    fpr_subj = 0.0001
    fs = 2000
    N = 16
    L = [1, 2, 5, 10]
    compare_k0 = 3.25


    fft_dirnames = ['/home/hgeng4/THESIS/results/Fmethod_fft/detection_ml/phi_0.7853981633974483/N_16/L_' + str(l) for l
                    in L]
    fht_dirnames = ['/home/hgeng4/THESIS/results/Fmethod_fht/detection_ml/phi_0.7853981633974483/N_16/L_' + str(l) for l
                    in L]
    jdht_dirnames = [
        '/home/hgeng4/THESIS/results/Fmethod_fht_jitter/detection_ml/phi_0.7853981633974483/N_16/L_' + str(l) for l in
        L]
    ddht_dirnames = [
        '/home/hgeng4/THESIS/results/Fmethod_fht_ditter/detection_ml/phi_0.7853981633974483/N_16/L_' + str(l) for l in
        L]

    compare_dirs = fht_dirnames
    compare_kernels = 'fht'
    legend_name = 'DHT'

    dft_datas, dht_datas = [], []
    fft_noise_power = np.load('fft_noise_power.npy', allow_pickle=True) if os.path.exists('fft_noise_power.npy') else None
    fht_noise_power = np.load('fht_noise_power.npy', allow_pickle=True) if os.path.exists('fht_noise_power.npy') else None
    jdht_noise_power = np.load('jdht_noise_power.npy', allow_pickle=True) if os.path.exists('jdht_noise_power.npy') else None
    ddht_noise_power = np.load('ddht_noise_power.npy', allow_pickle=True) if os.path.exists('ddht_noise_power.npy') else None

    for idx, l in enumerate(L):
        noise_levels_fft, tprs_fft, freqs_fft, out_power_fft = tp_vs_snr(fft_dirnames[idx], fpr_subj, 'fft')
        noise_levels_fht, tprs_fht, freqs_fht, out_power_fht = tp_vs_snr(compare_dirs[idx], fpr_subj, compare_kernels)
        # noise_levels_jdht, tprs_jdht, freqs_jdht, out_power_jdht = tp_vs_snr(jdht_dirnames[l], fpr_subj, 'fht_jitter')
        # noise_levels_ddht, tprs_ddht, freqs_ddht, out_power_ddht = tp_vs_snr(ddht_dirnames[l], fpr_subj, 'fht_ditter')

        dft_plot_data = PlotData(
            noise_levels=noise_levels_fft, tprs=tprs_fft, freqs=freqs_fft,
            compute_output_power=out_power_fft, how_often=25,
            fpr_subj=0.05, N=16, fs=2000, L=l, method='DFT'
        )

        dht_plot_data = PlotData(
            noise_levels=noise_levels_fht, tprs=tprs_fht, freqs=freqs_fht,
            compute_output_power=out_power_fht, how_often=25,
            fpr_subj=0.05, N=16, fs=2000, L=l, method=legend_name
        )

        dft_datas.append(dft_plot_data)
        dht_datas.append(dht_plot_data)

    # fft_noise_power = get_all_output_noise_power(L, noise_levels_fft, freqs_fft, N, 'fft') if fft_noise_power is None else fft_noise_power
    # fht_noise_power = get_all_output_noise_power(L, noise_levels_fht, freqs_fht, N, 'fht') if fht_noise_power is None else fht_noise_power

    # np.save('fft_noise_power.npy', fft_noise_power)
    # np.save('fht_noise_power.npy', fht_noise_power)
    # np.save('jdht_noise_power.npy', jdht_noise_power)
    # np.save('ddht_noise_power.npy', ddht_noise_power)


    compare_tpr(dft_datas, dht_datas, compare_k0)
    #compare_snrF(dft_datas, dht_datas, compare_k0, fft_noise_power, fht_noise_power)
