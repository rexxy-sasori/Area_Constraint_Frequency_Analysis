import numpy as np


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