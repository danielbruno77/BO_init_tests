# Initial-design comparison for Bayesian optimisation

Does the space-filling quality of the initial design matter for BO, and how many
initial points should we spend? This study runs the same GP / Expected
Improvement loop from four initial designs and three initial-sample sizes, on
Rastrigin and on two classic constrained engineering design problems.

## How to reproduce

```bash
pip install -r requirements.txt
python -m bo_bench.init_study.run       # full sweep, resumable, uses all cores
python -m bo_bench.init_study.report    # writes reports/init_study/{summary.md,*.png}
```

`run` accepts `--problems`, `--designs`, `--seeds` (count, default 20) and
`--workers`. Raw runs go to `results/init_study/raw/` (git-ignored), and the
report also writes one per-evaluation CSV per problem to `results/init_study/`.

## Setup

**Initial designs** (`bo_bench/init_study/designs.py`), all on the unit cube and
mapped to the problem bounds:

| name | method |
| --- | --- |
| `random` | i.i.d. uniform |
| `lhs` | Latin hypercube (`scipy.stats.qmc.LatinHypercube`) |
| `sobol` | scrambled Sobol (`scipy.stats.qmc.Sobol`); n is not forced to a power of 2 |
| `halton` | scrambled Halton (`scipy.stats.qmc.Halton`) |

**Initial-sample sizes**: n_init = 2d, 5d and 10d, counted inside a fixed budget
of 20d evaluations (so 10d spends half the budget on the design).

**Problems** (`bo_bench/init_study/problems.py`), all minimised:

| problem | d | constraints | budget | best known |
| --- | ---: | ---: | ---: | ---: |
| `rastrigin2d` | 2 | 0 | 40 | 0 |
| `rastrigin4d` | 4 | 0 | 80 | 0 |
| `pressure_vessel` (Sandgren 1990, continuous) | 4 | 3 | 80 | 5885.33 |
| `welded_beam` (Coello 2000 formulation) | 4 | 6 | 80 | 1.72485 |

The engineering constraints are written in normalised form (g / limit - 1 <= 0).

**Optimiser** (`bo_bench/init_study/bo.py`): a GP with a Matern-5/2 ARD kernel
(scikit-learn), Expected Improvement, and for the constrained problems one GP
per constraint, combined by multiplying EI with the probability of feasibility
(Gardner et al. 2014). Until a feasible point exists the acquisition maximises
the probability of feasibility alone. Constraints are treated as black boxes
evaluated together with the objective. The engineering objectives are modelled
in log space. The acquisition is maximised by random search plus three rounds
of local perturbation around the best candidates.

The harness's library adapters (`bayes_opt`, `skopt`) were not used because
neither supports black-box constraints through a common interface; using one
in-house loop also keeps the only difference between runs the initial design.

**Seeds**: 20 per configuration. For a given (problem, n_init, seed) every
design uses the same seed, for both the design and the BO loop.

**Metric**: best feasible objective so far after each evaluation. Plots show
the median and interquartile range over seeds of the regret (absolute for
Rastrigin, relative to the best known value for the engineering problems).
