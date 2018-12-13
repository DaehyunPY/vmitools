#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sympy import symbols, legendre, lambdify
from cytoolz import memoize, identity
from numpy import vectorize, cos, pi
from numpy.polynomial.legendre import legval
try:
    from numba import jit
except ImportError:
    print("Module 'numba' is not imported!")
    jit = identity


def __init_halfsummed_pn():
    n = symbols('n', integer=True, positive=True)
    expr = (legendre(n-1, 0)-legendre(n+1, 0))/(2*n+1)
    f = memoize(lambdify(n, expr, ('math', 'mpmath', 'numpy')))

    @vectorize
    def summed(n, region='top'):
        if n < 0:
            raise ValueError("Par 'n' is invalid!")
        elif n == 0:
            return 1.0
        elif n % 2 == 0:
            return 0.0
        if region == 'top':
            return f(n)
        elif region == 'btm':
            return -f(n)
        else:
            raise ValueError("Par 'region' have to be 'top' or 'btm'!")
    return summed
halfsummed_pn = __init_halfsummed_pn()


def full_asym(*betas):  # from 0th term
    if len(betas) == 0:
        raise ValueError("One or more args are needed!")

    fst, *tail = betas
    summed = 2.0 * fst

    n = len(tail)
    top = halfsummed_pn(range(1, n+1), 'top')
    btm = halfsummed_pn(range(1, n+1), 'btm')
    diffed = (top-btm).dot(tail)

    return diffed/summed
asym = full_asym


@jit
def simple_asym(*betas):
    top = legval(cos(0), betas)
    btm = legval(cos(pi), betas)
    return (top-btm)/(top+btm)
