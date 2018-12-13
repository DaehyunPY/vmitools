#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from textwrap import dedent
from operator import add

from cytoolz import reduce
from numpy import sin, cos, matrix, array, zeros, tensordot
from scipy.optimize import fmin
from numba import jit

from .coordinate import convert_xy2rth


@jit
def get_rotation_mat(th):
    return matrix(((cos(th), -sin(th)),
                   (sin(th), cos(th))))


@jit
def get_horshear_mat(m):
    return matrix(((1, m),
                   (0, 1)))


@jit
def get_squeeze_mat(k):
    return matrix(((k, 0),
                   (0, 1/k)))


class LinearTransformer:
    def __init__(self, mat, o=None):
        self.__mat = matrix(mat, dtype='float64')
        n, n1 = self.mat.shape
        if n != n1:
            raise ValueError('The matrix must be square!')
        if o is None:
            o = zeros(n)
        self.__o = array(o, dtype='float64')
        n2, *_ = self.o.shape
        if n != n2:
            raise ValueError(dedent(
                    """\
                    The shapes of hist and edges do not match each others!"""))

    @property
    def mat(self):
        return array(self.__mat)

    @property
    def inv(self):
        return array(self.__mat.I)

    @property
    def o(self):
        return self.__o

    def transform(self, *x):
        return tensordot(self.mat, tuple(x1-x0 for x0, x1 in zip(self.o, x)), axes=((1,), (0,)))

    def invert(self, *x):
        return (x1+x0 for x0, x1 in zip(self.o, tensordot(self.inv, x, axes=((1,), (0,)))))

    def __call__(self, *args):
        return self.transform(*args)

    @property
    def transformer(self):
        return self.transform

    @property
    def inverter(self):
        return self.invert

    @property
    def operators(self):
        return self.transformer, self.inverter


# high level objects
class SqueezeTransformer(LinearTransformer):
    def __init__(self, ph=0, m=0, k=1, th=0, x0=0, y0=0):
        rot1 = get_rotation_mat(th)
        squ = get_squeeze_mat(k)
        she = get_horshear_mat(m)
        rot2 = get_rotation_mat(-th)
        rot3 = get_rotation_mat(ph)
        mat = rot3.dot(rot2).dot(she).dot(squ).dot(rot1)
        super().__init__(mat, o=(x0, y0))


def opt_squ_pars(*smp, **init):
    """
    Optimize parameters to make the sample dots symmetric
    """
    keys = init.keys()

    def norm(pars):
        transformer = SqueezeTransformer(**dict(zip(keys, pars)))
        r, _ = convert_xy2rth(*transformer(*smp))
        return r.std()
    print('Optimizing parameters of squeeze transformation...')
    ret = fmin(norm, [init[k] for k in keys])
    print('Optimized parameters are...')
    report = reduce(add, ('    '+key+': {'+key+'}\n' for key in keys))
    print(report.format(**dict(zip(keys, ret))))
    return dict(zip(keys, ret))
