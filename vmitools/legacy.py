#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pandas import read_fwf
from numpy import zeros, array, ones


# with pre-calculated matrix
def get_reconstructer_with_mat(mat):
    nr, np, _, _ = mat.shape

    def transformer(
            r_dist,  # shape=(r,p)
            r):  # shape=(r)
        x_dist = r_dist/r[:, None]  # shape=(r,p)
        return mat.reshape(nr, np, nr*np).dot(x_dist.reshape(nr*np))

    def inverter(
            reconstructed,  # shape=(r,p)
            r,  # shape=(r)
            basis):  # shape=(p,th)
        _, nth = basis.shape
        img = zeros((nr, nth))  # shape=(r,th)
        coeff_rsq = (reconstructed*r[:, None]).T  # shape=(p,r)
        b = ones((1, nth))  # shape=(1,th)
        for i in range(nr):
            c = coeff_rsq[:, i:i+1].dot(b)  # shape=(p,th)
            img[i, :] = (c*basis).sum(0)  # shape=(th)
        return img
    return transformer, inverter


def get_reconstructer_with_matfile(filename, shape, header=None, **kwargs):
    with open(filename) as f:
        mat = array(read_fwf(f, header=header, **kwargs)).reshape(shape*2)
    return get_reconstructer_with_mat(mat)
