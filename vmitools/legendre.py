from typing import Tuple, Callable

from numpy import array, append, ndarray
from sympy import symbols, legendre, lambdify, integrate, cos
from cytoolz import memoize


__all__ = [
    'finite_legendre_transform_in_theta',
]


@memoize
def __init_basis(n: int) -> Callable[[ndarray, ndarray], ndarray]:
    th, fr, to = symbols('theta, from, to', real=True)
    expr = integrate(legendre(n, cos(th)), (th, fr, to))
    return lambdify((fr, to), expr, 'numpy')


def __get_basis(edges, n: int) -> ndarray:
    lambdified = (__init_basis(i) for i in range(n))
    return array([f(edges[:-1], edges[1:]) for f in lambdified])


@memoize
def __init_inversed(n: int) -> Callable[[ndarray, ndarray], ndarray]:
    x, fr, to = symbols('x, from, to', real=True)
    expr = integrate(legendre(n, x), (x, cos(fr), cos(to))) * (2 * n + 1) / 2
    return lambdify((fr, to), expr, 'numpy')


def __get_inversed(edges: ndarray, n: int) -> ndarray:
    lambdified = (__init_inversed(i) for i in range(n))
    return array([f(edges[:-1], edges[1:]) for f in lambdified])


def __finite_legendre_transform_in_theta(
    hist: ndarray,  # shape=(x,)
    edges: ndarray,
    n: int,
) -> Tuple[ndarray, ndarray]:
    inversed = __get_inversed(edges, n)  # shape=(n,x)
    basis = __get_basis(edges, n)
    dim = len(hist.shape)
    norm = (edges[1:] - edges[:-1]).reshape((-1,) + (dim - 1) * (1,))
    coeff = inversed.dot(hist / norm)
    return coeff, basis


def finite_legendre_transform_in_theta(
    hist: ndarray,
    edges: ndarray,
    n: int,
) -> Tuple[ndarray, ndarray, ndarray]:
    lefts = edges[:-1]
    rights = edges[1:]
    idx: ndarray = lefts < 0

    def sliced(idx: ndarray) -> Tuple[ndarray, ndarray]:
        return hist[idx], append(lefts[idx], rights[idx][-1])

    coeff_neg, basis_neg = __finite_legendre_transform_in_theta(
        *sliced(idx),
        n=n,
    )
    coeff_pos, basis_pos = __finite_legendre_transform_in_theta(
        *sliced(~idx),
        n=n,
    )
    return coeff_neg, -coeff_pos, append(basis_neg, basis_pos, axis=1)
