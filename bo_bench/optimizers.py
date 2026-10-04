"""Adapters that run each BO library on an Objective with a fixed budget.

Each adapter returns the sequence of objective values (maximisation
convention) in the order they were evaluated, exactly ``budget`` long.
"""

from typing import Callable, Dict, List

from .objectives import Objective


class _Recorder:
    """Wraps an objective and records every evaluation, in maximisation convention."""

    def __init__(self, objective: Objective, budget: int):
        self.objective = objective
        self.budget = budget
        self.values: List[float] = []

    def __call__(self, x) -> float:
        value = self.objective(x)
        self.values.append(value)
        return value


def run_bayes_opt(objective: Objective, budget: int, n_init: int, seed: int) -> List[float]:
    from bayes_opt import BayesianOptimization

    rec = _Recorder(objective, budget)
    names = [f"x{i}" for i in range(objective.dim)]
    optimizer = BayesianOptimization(
        f=lambda **kw: rec([kw[n] for n in names]),
        pbounds=dict(zip(names, objective.bounds)),
        random_state=seed,
        verbose=0,
        allow_duplicate_points=True,
    )
    optimizer.maximize(init_points=n_init, n_iter=budget - n_init)
    return rec.values[:budget]


def run_skopt(objective: Objective, budget: int, n_init: int, seed: int) -> List[float]:
    from skopt import gp_minimize

    rec = _Recorder(objective, budget)
    # skopt minimises, so hand it the negated objective (i.e. positive Rastrigin).
    gp_minimize(
        lambda x: -rec(x),
        dimensions=objective.bounds,
        n_calls=budget,
        n_initial_points=n_init,
        initial_point_generator="random",
        acq_func="EI",
        random_state=seed,
    )
    return rec.values[:budget]


OPTIMIZERS: Dict[str, Callable[[Objective, int, int, int], List[float]]] = {
    "bayes_opt": run_bayes_opt,
    "skopt": run_skopt,
}


def run_optimizer(method: str, objective: Objective, budget: int, n_init: int, seed: int) -> List[float]:
    if method not in OPTIMIZERS:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(OPTIMIZERS)}")
    if not 0 < n_init <= budget:
        raise ValueError("n_init must be between 1 and budget")
    values = OPTIMIZERS[method](objective, budget, n_init, seed)
    if len(values) != budget:
        raise RuntimeError(f"{method} made {len(values)} evaluations, expected {budget}")
    return values
