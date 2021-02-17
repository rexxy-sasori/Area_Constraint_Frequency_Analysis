import numpy as np
from scipy.signal import butter, lfilter

pi = np.pi


def reformat(arr, block_size, hop_size, num_sample):
    def index_func(i):
        return hop_size * i, hop_size * i + block_size

    arr = np.array([
        arr[index_func(i)[0]: index_func(i)[1]] for i in range(num_sample)
        if index_func(i)[1] <= num_sample
    ])

    return arr


def snr(num, denom):
    return 10 * np.log10(num / denom)


def bin_occurr(arr):
    val_occur = {}
    for val in arr:
        if val_occur.get(val) is None:
            val_occur[val] = 1
        else:
            val_occur[val] += 1

    vals = np.array(sorted(val_occur.keys()))
    occurs = np.array([val_occur.get(val) for val in vals])

    return vals, occurs


def mode(ndarray, axis=0):
    # Check inputs
    ndarray = np.asarray(ndarray)
    ndim = ndarray.ndim
    if ndarray.size == 1:
        return (ndarray[0], 1)
    elif ndarray.size == 0:
        raise Exception('Cannot compute mode on empty array')
    try:
        axis = range(ndarray.ndim)[axis]
    except:
        raise Exception('Axis "{}" incompatible with the {}-dimension array'.format(axis, ndim))

    # If array is 1-D and np version is > 1.9 np.unique will suffice
    if all([ndim == 1,
            int(np.__version__.split('.')[0]) >= 1,
            int(np.__version__.split('.')[1]) >= 9]):
        modals, counts = np.unique(ndarray, return_counts=True)
        index = np.argmax(counts)
        return modals[index], counts[index]

    # Sort array
    sort = np.sort(ndarray, axis=axis)
    # Create array to transpose along the axis and get padding shape
    transpose = np.roll(np.arange(ndim)[::-1], axis)
    shape = list(sort.shape)
    shape[axis] = 1
    # Create a boolean array along strides of unique values
    strides = np.concatenate([np.zeros(shape=shape, dtype='bool'),
                              np.diff(sort, axis=axis) == 0,
                              np.zeros(shape=shape, dtype='bool')],
                             axis=axis).transpose(transpose).ravel()
    # Count the stride lengths
    counts = np.cumsum(strides)
    counts[~strides] = np.concatenate([[0], np.diff(counts[~strides])])
    counts[strides] = 0
    # Get shape of padded counts and slice to return to the original shape
    shape = np.array(sort.shape)
    shape[axis] += 1
    shape = shape[transpose]
    slices = [slice(None)] * ndim
    slices[axis] = slice(1, None)
    # Reshape and compute final counts
    counts = counts.reshape(shape).transpose(transpose)[slices] + 1

    # Find maximum counts and return modals/counts
    slices = [slice(None, i) for i in sort.shape]
    del slices[axis]
    index = np.ogrid[slices]
    index.insert(axis, np.argmax(counts, axis=axis))
    return sort[index], counts[index]


class Msg:
    def __init__(self, content):
        self.content = content


def butter_low(cutoff, fs, order=5):
    nyq = 0.5 * fs
    ncutoff = cutoff / nyq
    b, a = butter(order, ncutoff, btype='low')
    return b, a


def filter_data(x, cutoff, fs):
    b, a = butter_low(cutoff, fs)
    y = lfilter(b, a, x)
    return y


def dft_output_signal_power(freq_o, phi, fs=2000, N=16,L=1):
    omega_o = 2 * pi * freq_o / fs

    def round_idx(float):
        floor = np.floor(float)
        ceil = np.ceil(float)

        if float - floor >= ceil - float:
            return int(ceil)
        else:
            return int(floor)

    def get_left_pulse(omega_o, bin_idx, N):
        if omega_o == 2 * pi * bin_idx / N:
            return 0

        left_center = omega_o + 2 * pi * bin_idx / N
        mag = np.sin(N * left_center / 2) / np.sin(left_center / 2) / 2
        sqmag = mag ** 2

        return sqmag

    def get_right_pulse(omega_o, bin_idx, N):
        if omega_o == 2 * pi * bin_idx / N:
            return (N / 2) ** 2

        right_center = omega_o - 2 * pi * bin_idx / N
        mag = np.sin(N * right_center / 2) / np.sin(right_center / 2) / 2
        sqmag = mag ** 2

        return sqmag

    def get_cross(omega_o, bin_idx, N, phi, l):
        if omega_o == 2 * pi * bin_idx / N:
            return 0

        factor_num = 1 - np.cos(N * omega_o)
        factor_denom = 2 * (np.cos(2 * pi * bin_idx / N) - np.cos(omega_o))
        factor = factor_num - factor_denom
        return factor * np.cos((N - 1) * omega_o + 2 * phi + 2*l*N)

    bin_idx = round_idx(freq_o * N / fs)
    left = get_left_pulse(omega_o, bin_idx, N)
    right = get_right_pulse(omega_o, bin_idx, N)

    cross = np.mean(np.array([get_cross(omega_o, bin_idx, N, phi, l) for l in range(L)]))

    return (left + right + cross) / N ** 2
