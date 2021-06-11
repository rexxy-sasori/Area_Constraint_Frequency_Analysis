import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from numba import njit

from IO import config
from core import freq_transform
from core import signal_generator
from core import utils
from core.utils import MILLISECONDS_IN_SECONDS

import scipy

color = [
   'b',
   'g',
   'r',
   'c',
   'm',
   'y',
   'k'
]


def unisign_quant(data, bits, clip, quant_flag):
    if not quant_flag:
        return data

    BW = bits
    W = np.clip(data, -clip, clip)
    W = W / clip
    Wq = np.minimum(
        np.round(W * np.power(2.0, BW - 1.0)) * np.power(2.0, 1.0 - BW),
        1.0 - np.power(2.0, 1.0 - BW)
    )

    Wq = clip * Wq
    return Wq


def quant_routine(data, n_bits, clip, quant_flag=False):
    if data.dtype == np.complex128:
        real_q = unisign_quant(np.real(data), n_bits, clip, quant_flag)
        imag_q = unisign_quant(np.imag(data), n_bits, clip, quant_flag)
        return real_q + 1j * imag_q
    else:
        return unisign_quant(data, n_bits, clip, quant_flag)


def get_min(data):
    if data.dtype == np.complex128:
        return np.min(np.abs(np.real(data))) + 1j * np.min(np.abs(np.imag(data)))
    else:
        return np.min(np.abs(data))


def get_max(data):
    ret = np.max(np.abs(data))
    return ret


def linear2log_safe(a, b):
    lamb = 1e-9
    return 10 * np.log10(a / (b + lamb)), a / (b + lamb)


def linear2log(ratio):
    return 10 * np.log10(ratio)


def dft_weight(N):
    return np.fft.fft(np.eye(N)) / N


def sig_second_moment(sig):
    return (np.abs(sig) ** 2).mean()


def get_real_quantization_noise_var(delta):
    return delta ** 2 / 12


def get_complex_quantization_noise_var(delta):
    return delta ** 2 / 6


def get_formula_error(predicted, simulated):
    predict_error_db, _ = linear2log_safe(predicted, simlated)
    return predict_error_db


def get_bin_idx(fo, fs=2000, N=16):
    ko = fo * N / fs
    ko_ceil = np.ceil(ko)
    ko_floor = np.floor(ko)

    if ko >= ko_floor + 0.5:
        return int(ko_ceil)
    else:
        return int(ko_floor)


def conj_transpose(matrix):
    return np.conjugate(matrix.T)


def sig_pow(sig):
    return np.square(np.abs(sig)).mean()


def compute_new_snr(orig_snr, sqnr):
    numerator = orig_snr / (1 + orig_snr)
    bottom = 1 / (1 + orig_snr) + 1 / sqnr

    return numerator / bottom


@njit
def unisign_quant_numba(data, bits, clip, quant_flag):
    if not quant_flag:
        return data

    data = np.maximum(np.minimum(data, clip), -clip)
    normalized = data / clip
    step_size = np.power(2.0, 1.0 - bits)
    num_level = np.round_(normalized / step_size)
    Wq = np.minimum(num_level * step_size, 1.0 - step_size)

    Wq = clip * Wq
    return Wq

@njit
def unisign_quant_numba_complex(data, bits, clip, quant_flag):
    real_quant = unisign_quant_numba(np.real(data), bits, clip, quant_flag)
    imag_quant = unisign_quant_numba(np.imag(data), bits, clip, quant_flag)

    return real_quant + 1j * imag_quant


@njit
def complex_dp_numba(seq, dp_num, dp_dim, coeff, qps=False, bps=0, cps=0):
    result = np.zeros(dp_num).astype(np.complex128)
    for dpidx in range(dp_num):
        total = 0
        for dimidx in range(dp_dim):
            add = coeff[dpidx, dimidx] * seq[dimidx]
            total = total + add
            total = unisign_quant_numba_complex(total, bps, cps, qps)

        result[dpidx] = total

    return result


logfile = 'myapp_dft.log'
if os.path.exists(logfile):
    os.remove(logfile)
logging.basicConfig(filename=logfile, level=logging.INFO)

input_quant = 1
weight_quant = 1
acc_quant = 1

noise_levels = [-20, -17.5,-15, -12.5, -10]
PREDS = []
SIMDS = []
IN_SNR = []
for nl in noise_levels:
    bs = []
    pred_snr = []
    sim_snr = []
    for b in range(3, 4):
        Bx = 6
        Bw = 5
        Ba = 8

        noise_level = nl
        noise_level_db = noise_level
        noise_level_linear = utils.db_to_linear(noise_level_db)
        f = 437.5

        configurations = config.parse_config('./yaml/example.yaml')
        configurations.signal.num_pos_decision = 10000
        configurations.signal.amps = [1]
        configurations.signal.freqs = [f]
        configurations.signal.phase = [0]

        configurations.noise.init_args.top = noise_level_linear
        configurations.noise.init_args.steady_state = noise_level_linear
        generator = signal_generator.InputSignalGenerator(configurations.signal, configurations.noise)
        signal_with_noise, _ = generator.get()

        configurations.noise.init_args.top = 0
        configurations.noise.init_args.steady_state = 0
        generator = signal_generator.InputSignalGenerator(configurations.signal, configurations.noise)
        signal_without_noise, _ = generator.get()
        noise_sample = signal_with_noise - signal_without_noise

        Nd = configurations.signal.num_pos_decision
        N = configurations.signal.block_size
        fs = configurations.signal.fs
        bin_idx = get_bin_idx(f, fs, N)

        signal_with_noise = signal_with_noise[0:Nd, :, :]
        signal_without_noise = signal_without_noise[0:Nd, :, :]
        noise_sample = noise_sample[0:Nd, :, :]

        coeff = scipy.linalg.dft(N, 'n')

        wmax = get_max(coeff)
        qcoeff = quant_routine(coeff, n_bits=Bw, clip=wmax, quant_flag=weight_quant)
        deltaw = qcoeff - coeff
        wsqnorm = np.real(coeff.dot(conj_transpose(coeff)).diagonal())
        deltawnorm = np.real(deltaw.dot(conj_transpose(deltaw)).diagonal())

        signal_max = 4 * signal_with_noise.std()
        # signal_max = get_max(signal_with_noise)
        qsignal_with_noise = quant_routine(signal_with_noise, Bx, signal_max, input_quant)
        q_input_noise = qsignal_with_noise - signal_with_noise

        deltax = utils.get_delta(Bx, signal_max)

        result_signal_without_noise = signal_without_noise.squeeze(1).dot(coeff.T)
        result_signal_without_noise = np.expand_dims(result_signal_without_noise, 1)
        result_signal_with_noise = signal_with_noise.squeeze(1).dot(coeff.T)
        result_signal_with_noise = np.expand_dims(result_signal_with_noise, 1)
        result_max = get_max(result_signal_with_noise)
        deltay = utils.get_delta(Ba, result_max)

        qresult_signal_with_noise = np.zeros(signal_without_noise.shape) * 1j

        start = datetime.now()
        for i in range(Nd):
            qresult_signal_with_noise[i, :, :] = complex_dp_numba(
                qsignal_with_noise[i, :, :][0], N, N, qcoeff, acc_quant, Ba, result_max
            )

        end = datetime.now()
        print('total duration: {} milliseconds'.format((end - start).total_seconds() * MILLISECONDS_IN_SECONDS))

        final_noise = qresult_signal_with_noise - result_signal_without_noise
        total_noise_power = sig_pow(final_noise)
        total_quant_noise = qresult_signal_with_noise - result_signal_with_noise
        total_quant_noise_power = sig_pow(total_quant_noise)
        
        out_signal_power = sig_pow(result_signal_without_noise[:, :, bin_idx])
        in_signal_power = sig_pow(signal_without_noise)
        in_noise_power = sig_pow(noise_sample)
        orig_at_output = wsqnorm[bin_idx] * in_noise_power

        input_quant_at_output = wsqnorm[bin_idx] * get_real_quantization_noise_var(deltax) if input_quant else 0
        weight_quant_at_output = (deltawnorm * sig_pow(signal_with_noise)).mean() if weight_quant else 0
        acc_quant_at_output = (N-1) * get_complex_quantization_noise_var(deltay) if acc_quant else 0
        pred_quant_noise_power_breakdown = np.array(
            [
                input_quant_at_output,
                weight_quant_at_output,
                acc_quant_at_output
            ]
        )
        pred_quant_noise_total = pred_quant_noise_power_breakdown.sum()
        out_signal_power_noquant = sig_pow(result_signal_with_noise[:, :, bin_idx])

        pred_sqnr_db, pred_sqnr_lin = linear2log_safe(out_signal_power_noquant, pred_quant_noise_total)
        sim_sqnr_db, sim_sqnr_lin = linear2log_safe(out_signal_power_noquant, total_quant_noise_power)
        in_snr_db, in_snr_lin = linear2log_safe(in_signal_power, in_noise_power)
        orig_out_snr_db, orig_out_snr_lin = linear2log_safe(out_signal_power, orig_at_output)
        sim_out_snr_db, sim_out_snr_lin = linear2log_safe(out_signal_power, total_noise_power)
        pred_out_snr_db = linear2log(compute_new_snr(orig_out_snr_lin, pred_sqnr_lin))

        print('PRED SQNR: {} dB'.format(pred_sqnr_db))
        print('SIM SQNR: {} dB'.format(sim_sqnr_db))
        # print('IN SNR: {} dB'.format(in_snr_db))
        # print('ORIG OUT SNR: {} dB'.format(orig_out_snr_db))

        print('PRED OUT SNR: {} dB'.format(pred_out_snr_db))
        print('SIM OUT SNR: {} dB'.format(sim_out_snr_db))
        print()

        # collect plotting data
        bs.append(b)
        pred_snr.append(pred_out_snr_db)
        sim_snr.append(sim_out_snr_db)

    PREDS.append(pred_snr)
    SIMDS.append(sim_snr)
    IN_SNR.append(in_snr_db)

legend_elements = [
    Line2D([0], [0], marker='o', color='k', label='Evaluation', markerfacecolor='k', markersize=8,linestyle='-'),
    Line2D([0], [0], marker='s', color='k', label='Simulation', markerfacecolor='k', markersize=8,linestyle='--'),
]

fig, ax = plt.subplots(figsize=(10,7))
for idx, nl in enumerate(noise_levels):
    ax.plot(bs, PREDS[idx], marker='o', color=color[idx], markersize=12,linestyle='-')
    ax.plot(bs, SIMDS[idx], marker='s', color=color[idx], markersize=12,linestyle='--')
    legend_elements.append(Line2D([0], [0], color=color[idx], lw=4, label='Input SNR={} dB'.format(-nl)))

ax.grid()
ax.set_xlabel(r'$B_a$ (bits)', fontsize=20)
ax.set_ylabel(r'$SNR_F$ (dB)', fontsize=20)
ax.tick_params(axis='both', labelsize=20)
ax.set_xticks(np.arange(3, 12, 1))
plt.legend(handles=legend_elements, loc='lower right',fontsize=15)
plt.savefig('accumulation_only_dft.png')

