from cytoolz import reduce
from numpy import logical_or, average
from numba import jit


@jit
def __mask(fr, to, v):
    return (fr <= v) & (v <= to)


def get_mask(*regs):
    def mask(v):
        return reduce(logical_or, (__mask(*reg, v=v) for reg in regs))
    return mask


@jit
def get_avg(arr, axis=None, weights=None):
    avg = average(arr, axis=axis, weights=weights)
    var = average((arr-avg)**2, axis=axis, weights=weights)
    return avg, var**0.5
