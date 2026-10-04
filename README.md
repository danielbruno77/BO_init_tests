# BO_init_tests

Experiments comparing Bayesian Optimisation libraries (and, later, initialisation strategies) on test and engineering design problems.

- `BO_optimiser_test.ipynb`: the original exploratory notebook.
- `bo_bench/`: a small benchmark harness that runs every method with the same seeds and evaluation budget.

## Setup

```bash
pip install -r requirements.txt
```

## Running the benchmark

```bash
python -m bo_bench.run --objective rastrigin2d --budget 50 --n-init 8 --seeds 0 1 2 3 4
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--objective` | `rastrigin2d` | Objective to optimise (`rastrigin2d`, `rastrigin2d_narrow`) |
| `--methods` | all | Any of `bayes_opt`, `skopt` |
| `--seeds` | `0 1 2 3 4` | One run per method per seed |
| `--budget` | `50` | Total objective evaluations per run, initial points included |
| `--n-init` | `8` | Random initial points (part of the budget) |
| `--out` | `results` | Output directory |

Outputs, in `--out`:

- `<objective>_results.csv`: one row per evaluation with columns `objective, method, seed, iteration, value, best_so_far`.
- `<objective>_convergence.png`: median best-so-far per method across seeds, with an interquartile band.

All values use the maximisation convention (higher is better; Rastrigin's optimum is 0). `skopt` minimises, so it is given the negated objective (positive Rastrigin) and its results are flipped back before recording.

## Layout

- `bo_bench/objectives.py`: objectives with bounds and known optimum. Add a new one to `OBJECTIVES`.
- `bo_bench/optimizers.py`: one adapter per library. Each takes `(objective, budget, n_init, seed)` and returns the evaluated values in order. Add a new one to `OPTIMIZERS`.
- `bo_bench/run.py`: CLI runner that writes the CSV and plot.
- `bo_bench/plotting.py`: convergence plot.
