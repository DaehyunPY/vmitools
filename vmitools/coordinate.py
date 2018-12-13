from typing import Tuple, Callable, Union

from numpy import sin, cos, arctan2, ndarray
from numba import jit


__all__ = [
    'convert_xy2rth',
    'convert_rth2xy',
    'convert_df_xy2rth',
    'convert_df_rth2xy',
]


@jit
def convert_xy2rth(
    x: Union[float, ndarray],
    y: Union[float, ndarray],
    x0: Union[float, ndarray] = 0,
    y0: Union[float, ndarray] = 0,
) -> Union[Tuple[float, float], Tuple[ndarray, ndarray]]:
    return ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5, arctan2(y - y0, x - x0)


@jit
def convert_rth2xy(
    r: Union[float, ndarray],
    th: Union[float, ndarray],
    x0: Union[float, ndarray] = 0,
    y0: Union[float, ndarray] = 0,
) -> Union[Tuple[float, float], Tuple[ndarray, ndarray]]:
    return r * cos(th) + x0, r * sin(th) + y0


def convert_df_xy2rth(
    df_xy: Callable[[float, float], float],
) -> Callable[[float, float], float]:
    def df_rth(r: float, th: float) -> float:
        x, y = convert_rth2xy(r, th)
        return df_xy(x, y) * r
    return df_rth


def convert_df_rth2xy(
    df_rth: Callable[[float, float], float],
) -> Callable[[float, float], float]:
    def df_xy(x: float, y: float) -> float:
        r, th = convert_xy2rth(x, y)
        return df_rth(r, th) / r
    return df_xy
