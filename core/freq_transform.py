import numpy as np
from numba import njit

from core.utils import filter_data, uniform_sign_quant


@njit
def one_stage_goertzel_bin(seq, len_seq, coeff, qps=False, bps=0, cps=0):
    total = 0
    for idx in range(len_seq):
        total = coeff * total + seq[idx]
    total = coeff * total
    return total


@njit
def one_stage_goertzel(seq, coeffs, length=16, qps=False, bps=8, cps=1):
    result = np.zeros(length).astype(np.complex128)
    for idx in range(length):
        coeff = coeffs[idx]
        result[idx] = one_stage_goertzel_bin(seq, length, coeff, qps, bps, cps)

    return result


@njit
def two_stage_goertzel_bin(seq, len_seq, coeff, qps=False, bps=0, cps=0):
    total = np.zeros(2)
    A_k = np.zeros((2, 2))
    A_k[0][0] = 2 * np.real(coeff)
    A_k[0][1] = -1
    A_k[1][0] = 1
    A_k[1][1] = 0

    for idx in range(len_seq):
        first = A_k[0][0] * total[0] + A_k[0][1] * total[1] + seq[idx]
        second = A_k[1][0] * total[0] + A_k[1][1] * total[1]

        total[0] = first
        total[1] = second

    first = A_k[0][0] * total[0] + A_k[0][1] * total[1]
    second = A_k[1][0] * total[0] + A_k[1][1] * total[1]

    total[0] = first
    total[1] = second

    total = total[0] - np.conjugate(coeff) * total[1]
    return total


@njit
def two_stage_goertzel(seq, coeffs, length=16, qps=False, bps=8, cps=1):
    result = np.zeros(length).astype(np.complex128)
    for idx in range(length):
        coeff = coeffs[idx]
        result[idx] = two_stage_goertzel_bin(seq, length, coeff, qps, bps, cps)

    return result


@njit
def dft_direct_dp_bin(seq, len_seq, coeff, qps=False, bps=0, cps=0):
    total = 0
    for idx in range(len_seq):
        total = coeff[idx] * seq[idx] + total


    return total


@njit
def dft_direct_dp(seq, coeffs, length=16, qps=False, bps=8, cps=1):
    result = np.zeros(length).astype(np.complex128)
    for idx in range(length):
        coeff = coeffs[idx, :]
        result[idx] = dft_direct_dp_bin(seq, length, coeff, qps, bps, cps)

    return result


@njit
def dht_dft_bin(seq, length, coeff, qps=False, bps=0, cps=0):
    total = 0
    for idx in range(length):
        total = total + coeff[idx] * seq[idx]
        total = uniform_sign_quant(total, bps, cps, qps)
    return total


@njit
def dht_dft(seq, coeffs, length=16, qps=False, bps=8, cps=1):
    result = np.zeros(length)
    for idx in range(length):
        coeff = coeffs[idx, :]
        result[idx] = dht_dft_bin(seq, length, coeff, qps, bps, cps)

    dft_result = np.zeros(length).astype(np.complex128)
    dft_result[0] = result[0] + 0j
    for idx in range(1, length):
        real = 0.5 * (result[idx] + result[length - idx])
        imag = 0.5 * (result[length - idx] - result[idx])
        dft_result[idx] = real + 1j * imag
    return dft_result


@njit
def dht_direct_dp_bin(seq, len_seq, coeff, qps=False, bps=0, cps=0):
    total = 0
    for idx in range(len_seq):
        total = coeff[idx] * seq[idx] + total
        #total = uniform_sign_quant(total, bps, cps, qps)

    return total


@njit
def dht_direct_dp(seq, coeffs, length=16, qps=False, bps=8, cps=1):
    result = np.zeros(length)
    for idx in range(length):
        coeff = coeffs[idx, :]
        result[idx] = dht_direct_dp_bin(seq, length, coeff, qps, bps, cps)

    return result


@njit
def dht_direct_dp_jitter(seq, coeffs, length=16, qps=False, bps=8, cps=1):
    return dht_direct_dp(seq, coeffs, length, qps, bps, cps)


def fast_dht_jitter(seq, axis=-1):
    if axis == -1:
        L, N = seq.shape
        coeffs = dht_coeff_jitter(N)
        return dht_direct_dp(seq[0], coeffs, length=N)
    else:
        L, N = seq.shape
        fht_seq_jiter = np.zeros(seq.shape)
        for i in range(L):
            coeffs = dht_coeff_jitter(N)
            fht_seq_jiter[i, :] = coeffs.dot(seq[i, :])
        return fht_seq_jiter


def fast_dht_jitter_filter(seq, axis=-1):
    if axis == -1:
        L, N = seq.shape
        coeffs = dht_coeff_filter_jitter(N)
        return dht_direct_dp(seq[0], coeffs, length=N)
    else:
        L, N = seq.shape
        fht_seq_jiter = np.zeros(seq.shape)
        for i in range(L):
            coeffs = dht_coeff_filter_jitter(N)
            fht_seq_jiter[i, :] = coeffs.dot(seq[i, :])
        return fht_seq_jiter


def fast_dht_ditter(seq, axis=-1):
    if axis == -1:
        L, N = seq.shape
        coeffs = dht_coeff_ditter(N)
        return dht_direct_dp(seq[0], coeffs, length=N)
    else:
        L, N = seq.shape
        fht_seq_jiter = np.zeros(seq.shape)
        coeffs = dht_coeff_ditter(N)
        for i in range(L):
            fht_seq_jiter[i, :] = coeffs.dot(seq[i, :])
        return fht_seq_jiter


def fast_dht(seq, axis=-1):
    if axis == -1:
        fft_seq = np.fft.fft(seq)
    else:
        row, col = seq.shape
        fft_seq = np.zeros(seq.shape).astype(np.complex128)
        for i in range(row):
            fft_seq[i, :] = np.fft.fft(seq[i, :])

    dht = np.real(fft_seq) - np.imag(fft_seq)
    return dht


def fast_dht_gaussian(seq, axis=-1):
    if axis == -1:
        L, N = seq.shape
        coeffs = dht_coeff_gaussian(N)
        return dht_direct_dp(seq[0], coeffs, length=N)
    else:
        L, N = seq.shape
        fht_seq_jiter = np.zeros(seq.shape)
        for i in range(L):
            coeffs = dht_coeff_gaussian(N)
            fht_seq_jiter[i, :] = coeffs.dot(seq[i, :])
        return fht_seq_jiter


def transform_all(seqs, freq_transform_configs, signal_configs):
    transform_kernel = __FREQ_TRANSFORMS__.get(freq_transform_configs.name)

    if transform_kernel is None:
        raise NotImplementedError

    N = signal_configs.block_size
    L = signal_configs.num_blocks_avg

    ret = np.zeros((len(seqs), L, N))
    seq_transforms = []

    for idx, seq in enumerate(seqs):
        seq = seq.reshape(L, -1)
        if L > 1:
            seq_transformed = transform_kernel(seq, axis=1)
        else:
            seq_transformed = transform_kernel(seq)

        seq_transforms.append(seq_transformed)

        ret[idx, :, :] = (np.abs(seq_transformed) ** 2) / (N ** 2)

    seq_transforms = np.array(seq_transforms)
    return ret, seq_transforms


__FREQ_TRANSFORMS__ = {
    'one_stage_goertzel': one_stage_goertzel,
    'two_stage_goertzel': two_stage_goertzel,
    'dft_direct_dp': dft_direct_dp,
    'dht_dft': dht_dft_bin,
    'dht_direct_dp': dht_direct_dp,
    'fft': np.fft.fft,
    'fht': fast_dht,
    'fht_jitter': fast_dht_jitter,
    'fht_ditter': fast_dht_ditter,
    'fht_gaussian': fast_dht_gaussian,
    'fht_jitter_filter': fast_dht_jitter_filter
}


def dht_coeff(block_size):
    idx = np.arange(block_size)
    idx_matrix = np.outer(idx, idx)
    coeffs = np.sqrt(2) * np.cos(2 * np.pi * idx_matrix / block_size - np.pi / 4)
    return coeffs


def dht_coeff_jitter(block_size):
    idx = np.arange(block_size)
    idx_matrix = np.outer(idx, idx)
    random_phases = np.random.uniform(0, 2 * np.pi, size=(block_size, block_size))
    coeffs = np.sqrt(2) * np.cos(2 * np.pi * idx_matrix / block_size - np.pi / 4 + random_phases)

    return coeffs


def dht_coeff_filter_jitter(block_size):
    idx = np.arange(block_size)
    idx_matrix = np.outer(idx, idx)
    random_phases = np.random.uniform(0, 2 * np.pi, size=(block_size, block_size))

    filtered_random_phases = filter_data(random_phases.reshape(block_size * block_size), 1000 / 16, 2000)
    filtered_random_phases = filtered_random_phases.reshape(block_size, block_size)

    coeffs = np.sqrt(2) * np.cos(2 * np.pi * idx_matrix / block_size - np.pi / 4 + filtered_random_phases)

    return coeffs


def dht_coeff_ditter(block_size):
    idx = np.arange(block_size)
    idx_matrix = np.outer(idx, idx)
    ditter_phase = (1 - np.sin(np.pi * np.arange(block_size) + np.pi / 2)) * np.pi / 4
    ditter_phase = np.array(block_size * [ditter_phase])
    coeffs = np.sqrt(2) * np.cos(2 * np.pi * idx_matrix / block_size - np.pi / 4 + ditter_phase)
    return coeffs


def dht_coeff_gaussian(block_size):
    # idx = np.arange(block_size)
    # idx_matrix = np.outer(idx, idx)
    # ditter_phase = (1 - np.sin(np.pi * np.arange(block_size) + np.pi / 2)) * np.pi / 4
    # ditter_phase = np.array(block_size * [ditter_phase])
    # coeffs = np.sqrt(2) * np.cos(2 * np.pi * idx_matrix / block_size - np.pi / 4 + ditter_phase)

    coeffs = np.random.normal(0, np.sqrt(10), size=(block_size, block_size))
    return coeffs



__COEFFS__ = {
    one_stage_goertzel: None,
    two_stage_goertzel: None,
    dft_direct_dp: None,
    dht_direct_dp: dht_coeff,
    dht_dft: dht_coeff,
    np.fft.fft: None,
    fast_dht: None,
    dht_direct_dp_jitter: dht_coeff_jitter,

}
