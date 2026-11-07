"""The three BO implementations compared, behind one interface.

``run(problem, X0, budget, seed) -> (f, feasible, seconds)`` where ``X0`` is the
initial design on the unit cube (shared by every method for a given seed, so
only the model / acquisition / optimiser differ) and ``f`` holds the objective
value of every evaluation in order, initial points included.

* ``skopt``: ``skopt.gp_minimize`` as a typical user would call it (GP with
  Matern-5/2 ARD, EI, default acquisition optimiser). It has no constraint
  support, so on the engineering problems infeasible points get a penalty.
* ``sklearn_loop``: the in-house loop from the initial-design study
  (scikit-learn GP, EI x probability of feasibility).
* ``botorch``: BoTorch ``SingleTaskGP`` (Matern-5/2 ARD + priors, fitted by
  MLL with L-BFGS) with ``LogExpectedImprovement``, optimised with multi-start
  gradient ascent through autograd. Constraints get their own GPs and enter
  through ``LogConstrainedExpectedImprovement`` (the same EI x PoF idea as
  ``sklearn_loop``, implemented natively).
* ``botorch_q4``: the same model, proposing 4 points per GP fit with
  ``qLogExpectedImprovement`` (Monte Carlo, constraints via its
  ``constraints`` argument), as with 4 simulations running in parallel.
"""
import time
import warnings

import numpy as np

# skopt: penalty for infeasible points (in log-objective space for the
# engineering problems): worst value seen so far plus the total violation.
PENALTY_WEIGHT = 10.0


def _objective_for_model(problem, f):
    return np.log(f) if problem.log_objective else f


# ----------------------------------------------------------------------------- skopt
def run_skopt(problem, X0, budget, seed):
    from skopt import gp_minimize

    F, G = [], []
    worst = [-np.inf]

    def f_skopt(u):
        x = problem.scale(np.asarray(u, dtype=float))
        f = problem.objective(x)
        g = problem.constraints(x)
        F.append(f)
        G.append(g)
        y = float(_objective_for_model(problem, f))
        worst[0] = max(worst[0], y)
        viol = float(np.clip(g, 0, None).sum())
        if viol > 0:
            y = worst[0] + PENALTY_WEIGHT * viol
        return y

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Initial points are passed as x0 (evaluated by skopt), so n_initial_points=0.
        gp_minimize(f_skopt, dimensions=[(0.0, 1.0)] * problem.dim,
                    x0=[list(map(float, u)) for u in X0],
                    n_calls=budget, n_initial_points=0, acq_func="EI",
                    random_state=seed)
    secs = time.time() - t0
    g = np.array(G).reshape(len(F), problem.n_constraints)
    return np.array(F[:budget]), np.all(g <= 0, axis=1)[:budget], secs


# ---------------------------------------------------------------------- sklearn loop
def run_sklearn_loop(problem, X0, budget, seed):
    from ..init_study import bo as ib

    # run_bo builds its own design; feed it ours by temporarily registering it.
    from ..init_study import designs
    designs.DESIGNS["_given"] = lambda n, d, s: np.asarray(X0)
    try:
        t0 = time.time()
        f, feas = ib.run_bo(problem, "_given", len(X0), budget, seed)
        secs = time.time() - t0
    finally:
        designs.DESIGNS.pop("_given", None)
    return f, feas, secs


# --------------------------------------------------------------------------- botorch
def run_botorch(problem, X0, budget, seed, q=1):
    import torch
    from botorch.acquisition import LogExpectedImprovement, qLogExpectedImprovement
    from botorch.acquisition.analytic import LogConstrainedExpectedImprovement
    from botorch.sampling import SobolQMCNormalSampler
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import ModelListGP, SingleTaskGP
    from botorch.models.transforms import Standardize
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood

    torch.manual_seed(seed)
    dtype = torch.double
    d, m = problem.dim, problem.n_constraints
    bounds = torch.stack([torch.zeros(d, dtype=dtype), torch.ones(d, dtype=dtype)])

    U, F, G = [], [], []

    def evaluate(u):
        x = problem.scale(np.asarray(u))
        U.append(np.asarray(u))
        F.append(problem.objective(x))
        G.append(problem.constraints(x))

    t0 = time.time()
    for u in X0:
        evaluate(u)

    while len(F) < budget:
        X = torch.tensor(np.array(U), dtype=dtype)
        # BoTorch maximises, so model the negated (log-)objective.
        y = -torch.tensor(_objective_for_model(problem, np.array(F)), dtype=dtype).unsqueeze(-1)
        g = torch.tensor(np.array(G).reshape(len(F), m), dtype=dtype)
        feas = (g <= 0).all(dim=-1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if m == 0:
                model = SingleTaskGP(X, y, outcome_transform=Standardize(m=1))
                fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
                acqf = (LogExpectedImprovement(model, best_f=y.max()) if q == 1
                        else qLogExpectedImprovement(model, best_f=y.max()))
            else:
                models = [SingleTaskGP(X, y, outcome_transform=Standardize(m=1))]
                models += [SingleTaskGP(X, g[:, j:j + 1], outcome_transform=Standardize(m=1))
                           for j in range(m)]
                model = ModelListGP(*models)
                fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
                if feas.any():
                    best_f = y[feas].max()
                else:  # no feasible point yet: any feasible point is an improvement
                    best_f = y.min() - 1.0
                if q == 1:
                    # Analytic EI x PoF; ~40x faster than the MC version below.
                    acqf = LogConstrainedExpectedImprovement(
                        model, best_f=best_f, objective_index=0,
                        constraints={j + 1: (None, 0.0) for j in range(m)})
                else:
                    acqf = qLogExpectedImprovement(
                        model, best_f=best_f,
                        objective=_first_output(),
                        constraints=[_constraint(j + 1) for j in range(m)],
                        sampler=SobolQMCNormalSampler(torch.Size([128])),
                    )
            # The joint q-batch MC problem is expensive with many constraint GPs,
            # so it gets a lighter multi-start budget.
            cand, _ = optimize_acqf(acqf, bounds=bounds, q=q,
                                    num_restarts=10 if q == 1 or m == 0 else 5,
                                    raw_samples=512 if q == 1 or m == 0 else 256)
        for u in cand.detach().numpy()[: budget - len(F)]:
            evaluate(u)

    secs = time.time() - t0
    gg = np.array(G).reshape(len(F), m)
    return np.array(F), np.all(gg <= 0, axis=1), secs


def _first_output():
    from botorch.acquisition.objective import GenericMCObjective
    return GenericMCObjective(lambda Z, X=None: Z[..., 0])


def _constraint(j):
    # BoTorch convention: constraint callables are feasible when <= 0, same as ours.
    return lambda Z: Z[..., j]


def run_botorch_q4(problem, X0, budget, seed):
    """Batch variant: 4 points per GP fit, as you would with 4 parallel CFD runs."""
    return run_botorch(problem, X0, budget, seed, q=4)


METHODS = {
    "skopt": run_skopt,
    "sklearn_loop": run_sklearn_loop,
    "botorch": run_botorch,
    "botorch_q4": run_botorch_q4,
}
