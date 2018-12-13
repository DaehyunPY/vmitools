#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from numpy import sin, cos, arctan2
from cytoolz import identity
try:
    from numba import jit
except ImportError:
    print("Module 'numba' is not imported!")
    jit = identity


@jit
def convert_xy2rth(x, y, x0=0.0, y0=0.0):
    return ((x-x0)**2+(y-y0)**2)**0.5, arctan2(y-y0, x-x0)


@jit
def convert_rth2xy(r, th, x0=0.0, y0=0.0):
    return r*cos(th)+x0, r*sin(th)+y0


def convert_df_xy2rth(df_xy):
    def df_rth(r, th):
        x, y = convert_rth2xy(r, th)
        return df_xy(x, y) * r
    return df_rth


def convert_df_rth2xy(df_rth):
    def df_xy(x, y):
        r, th = convert_xy2rth(x, y)
        return df_rth(r, th) / r
    return df_xy
