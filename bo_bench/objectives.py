"""Benchmark objectives.

Every objective is expressed in the *maximisation* convention used by
``bayes_opt``: higher is better. Optimisers that minimise (e.g. ``skopt``)
are given the negated function by the adapter in ``optimizers.py``, so the
values recorded in results are always comparable.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Objective:
    name: str
    func: Callable[[np.ndarray], float]  # takes a 1-D array, returns a float to maximise
    bounds: List[Tuple[float, float]]
    optimum: float  # known global maximum, used for regret

    @property
    def dim(self) -> int:
        return len(self.bounds)

    def __call__(self, x) -> float:
        return float(self.func(np.asarray(x, dtype=float)))


def neg_rastrigin(x: np.ndarray, A: float = 10.0) -> float:
    """Negative Rastrigin function; global maximum 0.0 at x = 0.

    For 2-D this is the notebook's
    ``10 (cos 2πx + cos 2πy) - (x² + y²) - 20``.
    """
    return -(A * x.size + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


def make_rastrigin(dim: int = 2, half_width: float = 5.12) -> Objective:
    return Objective(
        name=f"rastrigin{dim}d",
        func=neg_rastrigin,
        bounds=[(-half_width, half_width)] * dim,
        optimum=0.0,
    )


OBJECTIVES: Dict[str, Callable[[], Objective]] = {
    "rastrigin2d": lambda: make_rastrigin(2),
    # Narrower domain used in the notebook's bayes_opt run.
    "rastrigin2d_narrow": lambda: make_rastrigin(2, half_width=2.12),
}


def get_objective(name: str) -> Objective:
    try:
        return OBJECTIVES[name]()
    except KeyError:
        raise ValueError(f"Unknown objective {name!r}; choose from {sorted(OBJECTIVES)}") from None
