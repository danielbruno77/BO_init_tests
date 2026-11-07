# skopt `gp_minimize` vs BoTorch

Same problems, budgets and seeds as the initial-design study, with every method
starting from the same Latin-hypercube design of 2d points (the best initial
size in that study). Only the model, acquisition function and its optimiser
differ between methods.

## How to reproduce

```bash
pip install -r requirements.txt      # adds torch + botorch
python -m bo_bench.botorch_compare.run                       # 20 seeds, all methods
python -m bo_bench.botorch_compare.report                    # writes reports/botorch_compare/
```

`run` accepts `--problems`, `--methods`, `--seeds` and `--workers`, and resumes
from `results/botorch_compare/raw/` if interrupted. The full sweep took about
3.5 hours on 4 CPU cores (CPU-only PyTorch); almost all of it is the batch
variant on the constrained problems.

## Methods (`bo_bench/botorch_compare/methods.py`)

| name | what it is |
| --- | --- |
| `skopt` | `skopt.gp_minimize(acq_func="EI")` with the initial design passed as `x0`. No constraint support, so infeasible points are scored as worst-seen value + 10 x total violation (log-objective space). |
| `sklearn_loop` | The in-house loop from the initial-design study: scikit-learn GP, EI x probability of feasibility, random-search acquisition optimiser. |
| `botorch` | `SingleTaskGP` (Matern-5/2 ARD with BoTorch's default priors), `LogExpectedImprovement`, or `LogConstrainedExpectedImprovement` with one GP per constraint; acquisition maximised by multi-start L-BFGS through autograd (`optimize_acqf`). |
| `botorch_q4` | Same model, 4 points per iteration from `qLogExpectedImprovement` (Monte Carlo, constraints passed natively), i.e. 4 simulations in parallel. 10 seeds. |

See `summary.md` for the tables and plots.
