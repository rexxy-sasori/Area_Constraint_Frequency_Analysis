import pyximport
pyximport.install()

from libc.math cimport round, pow
import cython
from libc.stdio cimport printf

def cy_rounding(double n):
    return round(n)

cdef double rounding(double n) nogil:
    return round(n)

cdef double clip_num(double a, double min_value, double max_value) nogil:
    return min(max(a, min_value), max_value)

cdef double uniform_sign_quant(double data, int n_bits, double clip, int flag) nogil:
    if flag == 0:
        printf("don't quantize ... return ...")
        return data

    cdef double clip_data
    cdef double step_size
    cdef double level
    cdef double ret

    clip_data = clip_num(data, -clip, clip)
    step_size = pow(2, 1-n_bits)
    printf("step size: %f\n", step_size)
    level = rounding(clip_data / (step_size * clip))
    ret = clip * min(step_size * level, 1-step_size)

    return ret

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_mvm(double[:,:] seq, double[:, :] coeffs, int length, int bps, int qps, double cps, double[:] out):
    cdef int row_idx, col_idx
    cdef double total

    with cython.nogil:
        for row_idx in range(length):
            total = 0
            for col_idx in range(length):
                total = total + uniform_sign_quant(coeffs[row_idx, col_idx] * seq[0,col_idx], bps, cps, qps)

            out[row_idx] = total
