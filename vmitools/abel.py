#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sympy import symbols, lambdify, sqrt, asin
from numpy import zeros, array, matrix, append


def __init_basis():
    r, r0, r1, x, x0, x1 = symbols('r, r0, r1, x, x0, x1')
    # integrate(integrate(2*r/sqrt(r**2 - x**2), r), x)
    expr0 = r**2*asin(x/r) + r*x*sqrt(1 - x**2/r**2)
    expr1 = expr0.subs(x, x1)-expr0.subs(x, x0)
    expr2 = expr1.subs(r, r1)-expr1.subs(r, r0)
    func2 = lambdify((x0, x1, r0, r1), expr2, 'numpy')
    expr3 = expr0.subs(x, r)-expr0.subs(x, r0)
    expr4 = expr3.subs(r, r1)-expr3.subs(r, r0)
    func4 = lambdify((r0, r1), expr4, 'numpy')
    return func2, func4
__f2, __f4 = __init_basis()


def __get_basis(edges):
    n = len(edges)-1
    mat = zeros((n, n))  # shape=(x,r)
    for i in range(n):
        mat[:i, i] = __f2(
                edges[:i], edges[1:i+1], edges[i], edges[i+1])
        mat[i, i] = __f4(edges[i], edges[i+1])
    return matrix(mat)


def __abel_inverse(
        hist,  # shape=(x,)
        edges):
    mat = __get_basis(edges)  # shape=(x,r)
    return array(mat.I.dot(hist))  # shape=(r,x)(x,)=(r,)


def abel_inverse(
        hist,  # shape=(x,)
        edges):
    lefts = edges[:-1]
    rights = edges[1:]
    idx = 0 < rights

    def sliced(idx):
        return hist[idx], append(lefts[idx], rights[idx][-1])

    def flipped(hist, edges):
        return hist[::-1], edges[::-1]

    neg = __abel_inverse(*flipped(*sliced(~idx)))
    pos = __abel_inverse(*sliced(idx))
    return append(neg[::-1], pos, axis=0)  # shape=(r,)
