from typing import List, Mapping
from itertools import zip_longest

import numpy as np
from numpy.random import normal, choice, rand
from numpy.polynomial.legendre import Legendre

from vmitools import (
    abel_inverse, finite_legendre_transform_in_theta,
    interp, tohist, transpose_to_drdomega, transpose_to_drdth,
)


# Generate dist
def init_dist(beta: List[float], n: int = 1001) -> Mapping[str, np.ndarray]:
    x = np.linspace(-1, 1, n)
    fx = Legendre([1, *beta])(x)
    return {
        'x': x,
        'px': fx / fx.sum(),
    }


beta = [1, 1.5, 0.5, -0.5]
dist = init_dist(beta)


# Generate samples
def init_smp(
    n: int, r: float, sigma: float, x: float, px: float,
) -> Mapping[str, np.ndarray]:
    r = normal(loc=r, scale=sigma, size=n)
    z = r * choice(x, p=px, size=n)
    phi = 2 * np.pi * rand(n)
    x = (r ** 2 - z ** 2) ** 0.5 * np.cos(phi)
    y = (r ** 2 - z ** 2) ** 0.5 * np.sin(phi)
    return {
        'x': x,
        'y': y,
        'z': z,
    }


smp = init_smp(n=1000000, r=8, sigma=1, **dist)

# Generate projected image from the samples
zedges = np.linspace(-16, 16, 321)
img, *_ = np.histogram2d(smp['x'], smp['z'], bins=[zedges, zedges])

# Abel invert the image
dz = zedges[1:] - zedges[:-1]
inverted = abel_inverse(img, zedges) * dz[None, :]  # Shape: (rho, z)


# Transpose to other spaces
sliced = interp(inverted, zedges, zedges)  # (rho, z) -> intensity
redges, thedges = np.linspace(0, 16, 801), np.linspace(-np.pi, np.pi, 721)
hist_indrdomega = tohist(
    transpose_to_drdomega(sliced), redges, thedges,
)  # (r, th) -> intensity
hist_indrdth = tohist(
    transpose_to_drdth(sliced), redges, thedges,
)  # (r, th) -> intensity
_, coeff, _ = finite_legendre_transform_in_theta(
    hist_indrdomega.sum(0), thedges, 6,
)  # Shapes of returns: (n, r) (n, r) (n, th)

print("Expected   Simulated")
print("--------   ---------")
for expected, simulated in zip_longest(beta, coeff[1:]/coeff[0]):
    print(f"{expected if expected else 0:>8.3f}   {simulated:>9.3f}")
