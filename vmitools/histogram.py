from typing import Optional, Callable, Tuple

from numpy import array, arange, zeros, meshgrid, outer, ndarray
from cytoolz import curry
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from numba import jit


__all__ = [
    'Hist1d',
    'Hist2d',
]


@jit
def get_centers(edges: ndarray) -> ndarray:
    return (edges[1:] + edges[:-1]) / 2.0


@jit
def get_diffs(edges: ndarray) -> ndarray:
    return edges[1:] - edges[:-1]


class Hist1d:
    def __init__(
            self,
            hist: Optional[ndarray] = None,
            edges: Optional[ndarray] = None,
    ):
        if hist is None:
            n = len(edges) - 1
            hist = zeros(n)
        self.__hist = array(hist, dtype='float64')
        n, = self.hist.shape
        if edges is None:
            edges = arange(n + 1)
        self.__edges = array(edges, dtype='float64')

        if n != len(self.edges) - 1:
            raise ValueError(
                'The shapes of hist and edges do not match each others!')

    @property
    def copy(self) -> 'Hist1d':
        return Hist1d(self.hist, self.edges)

    @property
    def hist(self) -> ndarray:
        return self.__hist

    @hist.setter
    def hist(self, h: ndarray):
        self.hist[:] = h

    @property
    def img(self) -> ndarray:
        return self.intensity(self.edges)

    @img.setter
    def img(self, i: ndarray):
        self.intensity = UnivariateSpline(self.edges, i)

    @property
    def edges(self) -> ndarray:
        return self.__edges

    @property
    def centers(self) -> ndarray:
        return get_centers(self.edges)

    @property
    def diffs(self) -> ndarray:
        return get_diffs(self.edges)

    @property
    def intensity(self) -> Callable[[ndarray], ndarray]:
        return UnivariateSpline(self.centers, self.hist / self.diffs)

    @intensity.setter
    def intensity(self, i: Callable[[ndarray], ndarray]):
        self.hist = i(self.centers) * self.diffs


class Hist2d:
    def __init__(
            self,
            hist: Optional[ndarray] = None,
            x_edges: Optional[ndarray] = None,
            y_edges: Optional[ndarray] = None,
    ):
        if hist is None:
            nx = len(x_edges) - 1
            ny = len(y_edges) - 1
            hist = zeros((nx, ny))
        self.__hist = array(hist, dtype='float64')
        nx, ny = self.hist.shape
        if x_edges is None:
            x_edges = arange(nx + 1)
        if y_edges is None:
            y_edges = arange(ny + 1)
        self.__x_edges = array(x_edges, dtype='float64')
        self.__y_edges = array(y_edges, dtype='float64')

        def fst(*arr):
            ret, *_ = arr
            return ret

        if (
                nx != fst(*self.x_edges.shape) - 1
                or ny != fst(*self.y_edges.shape) - 1
        ):
            raise ValueError(
                'The shapes of hist and edges do not match each others!')

    @property
    def copy(self) -> 'Hist2d':
        return Hist2d(self.hist, self.x_edges, self.y_edges)

    @property
    def hist(self) -> ndarray:
        return self.__hist

    @hist.setter
    def hist(self, h: ndarray):
        self.hist[:, :] = h

    @property
    def img(self) -> ndarray:
        x, y = meshgrid(self.x_edges, self.y_edges, indexing='ij')
        return self.intensity(x, y)

    @img.setter
    def img(self, i: ndarray):
        x, y = self.x_edges, self.y_edges
        spline = RectBivariateSpline(x, y, i)
        self.intensity = curry(spline, grid=False)

    @property
    def x_edges(self) -> ndarray:
        return self.__x_edges

    @property
    def x_centers(self) -> ndarray:
        return get_centers(self.x_edges)

    @property
    def x_diffs(self) -> ndarray:
        return get_diffs(self.x_edges)

    @property
    def y_edges(self) -> ndarray:
        return self.__y_edges

    @property
    def y_centers(self) -> ndarray:
        return get_centers(self.y_edges)

    @property
    def y_diffs(self) -> ndarray:
        return get_diffs(self.y_edges)

    @property
    def intensity(self) -> Callable[[ndarray, ndarray], ndarray]:
        x, y = self.x_centers, self.y_centers
        dx, dy = self.x_diffs, self.y_diffs
        spline = RectBivariateSpline(x, y, self.hist / outer(dx, dy))
        return curry(spline, grid=False)

    @intensity.setter
    def intensity(self, i: Callable[[ndarray, ndarray], ndarray]):
        x, y = meshgrid(self.x_centers, self.y_centers, indexing='ij')
        dx, dy = self.x_diffs, self.y_diffs
        self.hist = i(x, y) * outer(dx, dy)

    @property
    def meshed_xyz(self) -> Tuple[ndarray, ndarray, ndarray]:
        return self.x_edges, self.y_edges, self.hist.T

    @property
    def meshed_yxz(self) -> Tuple[ndarray, ndarray, ndarray]:
        return self.y_edges, self.x_edges, self.hist
