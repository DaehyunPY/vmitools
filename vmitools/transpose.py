"""
Transpose to other spaces. Here, we use below annotations:
    Spherical coordinates (r, th, phi),
    Cartesian coordinates (x, y, z) and they are related by:
        r ** 2 = x ** 2 + y ** 2 + z ** 2,
        z = r * cos(th),
        rho = r * sin(th),
        x = rho * cos(phi),
        y = rho * sin(phi).
"""
from typing import Callable, Optional
from functools import partial
from warnings import warn

from numpy import ndarray, sin, cos, abs, pi, meshgrid, array, einsum, stack
from numpy.linalg import inv
from scipy.interpolate import RectBivariateSpline


__all__ = [
    'interp',
    'tohist',
    'transpose_linearly',
    'transpose_to_drdomega',
    'transpose_to_drdth',
]


def interp(
        hist: ndarray,
        xedges: ndarray,
        yedges: ndarray,
    ) -> Callable[[ndarray, ndarray], ndarray]:
    x = (xedges[1:] + xedges[:-1]) / 2
    y = (yedges[1:] + yedges[:-1]) / 2
    dx = xedges[1:] - xedges[:-1]
    dy = yedges[1:] - yedges[:-1]
    f = RectBivariateSpline(x, y, hist / dx[:, None] / dy[None, :])
    return partial(f, grid=False)


def tohist(
        f: Callable[[ndarray, ndarray], ndarray],
        xedges: ndarray,
        yedges: ndarray,
    ) -> ndarray:
    x, y = meshgrid(
        (xedges[1:] + xedges[:-1]) / 2,
        (yedges[1:] + yedges[:-1]) / 2,
        indexing='ij',
    )
    dx = xedges[1:] - xedges[:-1]
    dy = yedges[1:] - yedges[:-1]
    return f(x, y) * dx[:, None] * dy[None, :]


def transpose_linearly(
        cart: Callable[..., ndarray],
        mat: ndarray,
        x0: Optional[ndarray] = None,
        u0: Optional[ndarray] = None,
    ) -> Callable[..., ndarray]:
    """
    Transpose a function f: x -> intensity, linearly,
    to a function g: u -> intensity, which is related by:
        g(u) = f(x),
        u - u0 = mat @ (x - x0).
    :param cart:
    :param mat:
    :param x0:
    :param u0:
    :return:
    """
    if u0 is None:
        u0 = array(0)[None]
    if x0 is None:
        x0 = array(0)[None]
    minv = inv(mat)
    warn(
        "The behavior of this function is changed after 201906.0!",
        UserWarning,
    )

    def ret(*u: ndarray) -> ndarray:
        n = len(u)
        shape = slice(None), *(n * [None])
        x = einsum("ij,j...->i...", minv, stack(u) - u0[shape])
        return cart(*(x + x0[shape]))
    return ret


def transpose_to_drdomega(
        cart: Callable[[ndarray, ndarray], ndarray],
    ) -> Callable[[ndarray, ndarray], ndarray]:
    """
    Transpose a function f: (z, rho) -> intensity, of which volume dvol
    is expressed as:
        dvol = f(z, rho) dz drho rho dphi,
    to a function g: (r, th) -> intensity, of which the volume is
    expressed as:
        dvol = g(r, th) dr domega,
    where domega = dcos(th) dphi.
    Note that this system is similar to Spherical coordination system,
    but it is not the same.
    """
    def ret(r: ndarray, th: ndarray) -> ndarray:
        z, rho = r * cos(th), r * sin(th)
        return cart(rho, z) * r ** 2
    return ret


def transpose_to_drdth(
        cart: Callable[[ndarray, ndarray], ndarray],
    ) -> Callable[[ndarray, ndarray], ndarray]:
    """
    Transpose a function f: (z, rho) -> intensity, of which volume dvol
    is expressed as:
        dvol = f(z, rho) dz drho rho dphi,
    to a function h: (r, th) -> intensity, of which the volume is
    expressed as:
        dvol = h(r, th) dr dth.
    """
    def ret(r: ndarray, th: ndarray) -> ndarray:
        z, rho = r * cos(th), r * sin(th)
        return cart(rho, z) * r ** 2 * abs(sin(th)) * pi
    return ret
