"""Benchmark problems, all posed as minimisation.

Each problem exposes ``bounds`` (d x 2), ``objective(x)`` and ``constraints(x)``
returning an array where values <= 0 are feasible (empty for unconstrained).
Constraints are written in normalised form (g / limit - 1) so they are O(1)
and well suited to GP modelling.
"""
from dataclasses import dataclass, field

import numpy as np

from ..objectives import neg_rastrigin


@dataclass
class Problem:
    name: str
    bounds: np.ndarray
    f_opt: float  # best known value, used only for reporting regret
    n_constraints: int = 0
    log_objective: bool = False  # model log(f) in the GP (f > 0 everywhere)
    budget: int = field(default=0)

    @property
    def dim(self):
        return self.bounds.shape[0]

    def scale(self, u):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + u * (hi - lo)

    def objective(self, x):
        raise NotImplementedError

    def constraints(self, x):
        return np.zeros(0)


class Rastrigin(Problem):
    def __init__(self, dim=2):
        super().__init__(f"rastrigin{dim}d", np.tile([-5.12, 5.12], (dim, 1)), 0.0,
                         budget=20 * dim)

    def objective(self, x):
        return -float(neg_rastrigin(x))


class PressureVessel(Problem):
    """Cylindrical pressure vessel (Sandgren 1990), continuous relaxation.

    x = (Ts shell thickness, Th head thickness, R inner radius, L length).
    Best known continuous optimum ~ 5885.33.
    """

    def __init__(self):
        super().__init__("pressure_vessel",
                         np.array([[0.0625, 6.1875], [0.0625, 6.1875], [10.0, 200.0], [10.0, 200.0]]),
                         5885.3328, n_constraints=3, log_objective=True, budget=80)

    def objective(self, x):
        ts, th, r, l = x
        return float(0.6224 * ts * r * l + 1.7781 * th * r**2
                     + 3.1661 * ts**2 * l + 19.84 * ts**2 * r)

    def constraints(self, x):
        ts, th, r, l = x
        return np.array([
            0.0193 * r / ts - 1.0,
            0.00954 * r / th - 1.0,
            1.0 - (np.pi * r**2 * l + 4.0 / 3.0 * np.pi * r**3) / 1296000.0,
        ])
        # The classic g4 (L <= 240) is always inactive with L <= 200.


class WeldedBeam(Problem):
    """Welded beam design (Ragsdell & Phillips 1976; Coello 2000 formulation).

    x = (h weld thickness, l weld length, t bar height, b bar thickness).
    Best known optimum ~ 1.724852. The g5 bound h >= 0.125 is folded into the box.
    """

    P, L, E, G = 6000.0, 14.0, 30e6, 12e6
    TAU_MAX, SIGMA_MAX, DELTA_MAX = 13600.0, 30000.0, 0.25

    def __init__(self):
        super().__init__("welded_beam",
                         np.array([[0.125, 2.0], [0.1, 10.0], [0.1, 10.0], [0.1, 2.0]]),
                         1.724852, n_constraints=6, log_objective=True, budget=80)

    def objective(self, x):
        h, l, t, b = x
        return float(1.10471 * h**2 * l + 0.04811 * t * b * (14.0 + l))

    def constraints(self, x):
        h, l, t, b = x
        P, L, E, G = self.P, self.L, self.E, self.G
        tau_p = P / (np.sqrt(2) * h * l)
        M = P * (L + l / 2)
        R = np.sqrt(l**2 / 4 + ((h + t) / 2) ** 2)
        J = 2 * (np.sqrt(2) * h * l * (l**2 / 12 + ((h + t) / 2) ** 2))
        tau_pp = M * R / J
        tau = np.sqrt(tau_p**2 + 2 * tau_p * tau_pp * l / (2 * R) + tau_pp**2)
        sigma = 6 * P * L / (b * t**2)
        delta = 4 * P * L**3 / (E * t**3 * b)
        pc = (4.013 * E * np.sqrt(t**2 * b**6 / 36) / L**2) * (1 - t / (2 * L) * np.sqrt(E / (4 * G)))
        return np.array([
            tau / self.TAU_MAX - 1.0,
            sigma / self.SIGMA_MAX - 1.0,
            (h - b) / b,
            (0.10471 * h**2 + 0.04811 * t * b * (14.0 + l)) / 5.0 - 1.0,
            delta / self.DELTA_MAX - 1.0,
            1.0 - pc / P,
        ])


PROBLEMS = {
    "rastrigin2d": lambda: Rastrigin(2),
    "rastrigin4d": lambda: Rastrigin(4),
    "pressure_vessel": PressureVessel,
    "welded_beam": WeldedBeam,
}
