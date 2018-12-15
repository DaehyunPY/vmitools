from numpy import ndarray, array, sin, cos


__all__ = [
    'mrot',
    'mhorshear',
    'msqueeze',
]


def mrot(th: float) -> ndarray:
    return array(
        [[cos(th), -sin(th)],
         [sin(th), cos(th)]],
    )


def mhorshear(m: float) -> ndarray:
    return array(
        [[1, m],
         [0, 1]],
    )


def msqueeze(k: float) -> ndarray:
    return array(
        [[k, 0],
         [0, 1/k]],
    )
