"""Initial-design samplers on the unit hypercube [0, 1]^d."""
import warnings

import numpy as np
from scipy.stats import qmc


def uniform(n, d, seed):
    return np.random.default_rng(seed).random((n, d))


def lhs(n, d, seed):
    return qmc.LatinHypercube(d=d, seed=seed).random(n)


def sobol(n, d, seed):
    # Scrambled Sobol. n need not be a power of 2; scipy warns about balance
    # properties in that case, which we accept so every design uses the same n.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return qmc.Sobol(d=d, scramble=True, seed=seed).random(n)


def halton(n, d, seed):
    return qmc.Halton(d=d, scramble=True, seed=seed).random(n)


DESIGNS = {"random": uniform, "lhs": lhs, "sobol": sobol, "halton": halton}


def initial_design(name, n, d, seed):
    return DESIGNS[name](n, d, seed)
