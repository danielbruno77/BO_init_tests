"""Minimal GP-based Bayesian optimisation with (constrained) Expected Improvement.

Search happens on the unit cube; the problem maps it to physical bounds.
Black-box constraints are modelled with independent GPs and combined with EI
through the probability of feasibility (Gardner et al. 2014). Before any
feasible point is found, the acquisition maximises the probability of
feasibility alone.
"""
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from .designs import initial_design


def _fit_gp(X, y, rng, warm_kernel=None, n_restarts=2):
    """Fit a Matern-5/2 ARD GP; warm-start hyperparameters from ``warm_kernel`` if given."""
    d = X.shape[1]
    kernel = warm_kernel or ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.full(d, 0.3), length_scale_bounds=(1e-2, 1e1), nu=2.5)
    gp = GaussianProcessRegressor(kernel, alpha=1e-6, normalize_y=True,
                                  n_restarts_optimizer=n_restarts,
                                  random_state=int(rng.integers(2**31)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, y)
    return gp


def _acquisition(obj_gp, con_gps, best):
    def acq(U):
        U = np.atleast_2d(U)
        log_pof = np.zeros(len(U))
        for gp in con_gps:
            mu, s = gp.predict(U, return_std=True)
            log_pof += norm.logcdf(-mu / np.maximum(s, 1e-9))
        if best is None:  # no feasible point yet
            return log_pof
        mu, s = obj_gp.predict(U, return_std=True)
        s = np.maximum(s, 1e-9)
        z = (best - mu) / s
        ei = s * (z * norm.cdf(z) + norm.pdf(z))
        return np.log(np.maximum(ei, 1e-300)) + log_pof
    return acq


def _maximise(acq, d, rng, n_candidates=3000, n_top=10, n_local=300, n_rounds=3):
    """Random search followed by a few rounds of shrinking Gaussian perturbations
    around the best candidates (batched, so each round is a single GP predict)."""
    C = rng.random((n_candidates, d))
    vals = acq(C)
    step = 0.1
    for _ in range(n_rounds):
        top = C[np.argsort(vals)[-n_top:]]
        L = np.clip(np.repeat(top, n_local // n_top, axis=0)
                    + step * rng.standard_normal((n_local // n_top * n_top, d)), 0.0, 1.0)
        C, vals = np.vstack([C, L]), np.concatenate([vals, acq(L)])
        step /= 3
    return C[np.argmax(vals)]


def run_bo(problem, design, n_init, budget, seed):
    """Return per-evaluation objective values and feasibility flags (length = budget)."""
    rng = np.random.default_rng(seed)
    d = problem.dim
    U = list(initial_design(design, n_init, d, seed))
    F, G = [], []

    def evaluate(u):
        x = problem.scale(np.asarray(u))
        F.append(problem.objective(x))
        G.append(problem.constraints(x))

    for u in U:
        evaluate(u)

    kernels = [None] * (1 + problem.n_constraints)
    while len(F) < budget:
        X = np.array(U)
        f = np.array(F)
        g = np.array(G).reshape(len(F), problem.n_constraints)
        feas = np.all(g <= 0, axis=1)
        y = np.log(f) if problem.log_objective else f
        # Full multi-start hyperparameter fit every 10th iteration, warm-started
        # single-start fits in between.
        full = (len(F) - n_init) % 10 == 0
        targets = [y] + [g[:, j] for j in range(problem.n_constraints)]
        gps = [_fit_gp(X, t, rng, None if full else k, 2 if full else 0)
               for t, k in zip(targets, kernels)]
        kernels = [gp.kernel_ for gp in gps]
        obj_gp, con_gps = gps[0], gps[1:]
        best = y[feas].min() if feas.any() else None
        u = _maximise(_acquisition(obj_gp, con_gps, best), d, rng)
        if np.min(np.linalg.norm(X - u, axis=1)) < 1e-6:  # avoid duplicates
            u = rng.random(d)
        U.append(u)
        evaluate(u)

    g = np.array(G).reshape(len(F), problem.n_constraints)
    return np.array(F), np.all(g <= 0, axis=1)
