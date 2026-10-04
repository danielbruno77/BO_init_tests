"""Run every method on an objective with fixed seeds and a shared budget.

Example::

    python -m bo_bench.run --objective rastrigin2d --budget 50 --n-init 8 --seeds 0 1 2 3 4

Writes ``<out>/<objective>_results.csv`` (one row per evaluation) and
``<out>/<objective>_convergence.png`` (median best-so-far with IQR band).
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from .objectives import OBJECTIVES, get_objective
from .optimizers import OPTIMIZERS, run_optimizer
from .plotting import plot_convergence

CSV_FIELDS = ["objective", "method", "seed", "iteration", "value", "best_so_far"]


def run_benchmark(objective_name, methods, seeds, budget, n_init, out_dir):
    objective = get_objective(objective_name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{objective.name}_results.csv"

    curves = {m: [] for m in methods}
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for method in methods:
            for seed in seeds:
                t0 = time.perf_counter()
                values = run_optimizer(method, objective, budget, n_init, seed)
                best = np.maximum.accumulate(values)
                curves[method].append(best)
                for i, (v, b) in enumerate(zip(values, best), start=1):
                    writer.writerow({
                        "objective": objective.name, "method": method, "seed": seed,
                        "iteration": i, "value": v, "best_so_far": b,
                    })
                print(f"{method:>10} seed={seed:<3} best={best[-1]:9.4f}  ({time.perf_counter() - t0:.1f}s)")

    png_path = out_dir / f"{objective.name}_convergence.png"
    plot_convergence(
        {m: np.array(c) for m, c in curves.items()},
        png_path,
        title=f"{objective.name}: best-so-far over {len(seeds)} seeds",
        n_init=n_init,
        optimum=objective.optimum,
    )
    print(f"Wrote {csv_path} and {png_path}")
    return csv_path, png_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objective", default="rastrigin2d", choices=sorted(OBJECTIVES))
    p.add_argument("--methods", nargs="+", default=list(OPTIMIZERS), choices=sorted(OPTIMIZERS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--budget", type=int, default=50, help="total objective evaluations per run")
    p.add_argument("--n-init", type=int, default=8, help="random initial points (part of the budget)")
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)
    run_benchmark(args.objective, args.methods, args.seeds, args.budget, args.n_init, args.out)


if __name__ == "__main__":
    main()
