from operator import add
from typing import Union, List, Callable, Tuple, Mapping

from cytoolz import reduce
from numpy import sin, cos, matrix, array, zeros, tensordot, ndarray
from scipy.optimize import fmin
from numba import jit

from .coordinate import convert_xy2rth


__all__ = [
    'LinearTransformer',
    'SqueezeTransformer',
    'opt_squ_pars',
]


@jit
def get_rotation_mat(th: float) -> matrix:
    return matrix(((cos(th), -sin(th)),
                   (sin(th), cos(th))))


@jit
def get_horshear_mat(m: float) -> matrix:
    return matrix(((1, m),
                   (0, 1)))


@jit
def get_squeeze_mat(k: float) -> matrix:
    return matrix(((k, 0),
                   (0, 1/k)))


class LinearTransformer:
    def __init__(self, mat: ndarray, o: Union[None, List, ndarray] = None):
        self.__mat = matrix(mat, dtype='float64')
        n, n1 = self.mat.shape
        if n != n1:
            raise ValueError('The matrix must be square!')
        if o is None:
            o = zeros(n)
        self.__o = array(o, dtype='float64')
        n2, *_ = self.o.shape
        if n != n2:
            raise ValueError(
                "The shapes of hist and edges do not match each others!")

    @property
    def mat(self) -> ndarray:
        return array(self.__mat)

    @property
    def inv(self) -> ndarray:
        return array(self.__mat.I)

    @property
    def o(self) -> ndarray:
        return self.__o

    def transform(self, *x: float) -> ndarray:
        return tensordot(
            self.mat,
            [x1-x0 for x0, x1 in zip(self.o, x)],
            axes=((1,), (0,)),
        )

    def invert(self, *x: float) -> ndarray:
        return array([
            x1 + x0
            for x0, x1
            in zip(self.o, tensordot(self.inv, x, axes=[[1], [0]]))
        ])

    def __call__(self, *args: float) -> ndarray:
        return self.transform(*args)

    @property
    def transformer(self) -> Callable[..., ndarray]:
        return self.transform

    @property
    def inverter(self) -> Callable[..., ndarray]:
        return self.invert

    @property
    def operators(self) -> Tuple[Callable[..., ndarray],
                                 Callable[..., ndarray]]:
        return self.transformer, self.inverter


class SqueezeTransformer(LinearTransformer):
    def __init__(
        self,
        ph: float = 0,
        m: float = 0,
        k: float = 1,
        th: float = 0,
        x0: float = 0,
        y0: float = 0,
    ):
        rot1 = get_rotation_mat(th)
        squ = get_squeeze_mat(k)
        she = get_horshear_mat(m)
        rot2 = get_rotation_mat(-th)
        rot3 = get_rotation_mat(ph)
        mat = rot3.dot(rot2).dot(she).dot(squ).dot(rot1)
        super().__init__(mat, o=[x0, y0])


def opt_squ_pars(*smp: float, **init: float) -> Mapping[str, float]:
    """
    Optimize parameters to make the sample dots symmetric
    """
    keys = sorted(init.keys())

    def norm(pars: List[float]) -> float:
        transformer = SqueezeTransformer(**dict(zip(keys, pars)))
        r, _ = convert_xy2rth(*transformer(*smp))
        return r.std()

    print('Optimizing parameters of squeeze transformation...')
    ret = fmin(norm, [init[k] for k in keys])
    print('Optimized parameters are...')
    report = reduce(add, ('    '+key+': {'+key+'}\n' for key in keys))
    print(report.format(**dict(zip(keys, ret))))
    return dict(zip(keys, ret))
