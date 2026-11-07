"""Run the initial-design comparison.

Every (problem, design, n_init, seed) run is stored as its own .npz under
results/init_study/raw/, so an interrupted sweep resumes where it stopped.

    python -m bo_bench.init_study.run                                  # full sweep
    python -m bo_bench.init_study.run --problems rastrigin2d --seeds 3 # quick check
"""
import os

# One BLAS thread per worker process; parallelism comes from the process pool.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import itertools
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np  # noqa: E402

from .bo import run_bo  # noqa: E402
from .designs import DESIGNS  # noqa: E402
from .problems import PROBLEMS  # noqa: E402

RAW = Path(__file__).resolve().parents[2] / "results" / "init_study" / "raw"
INIT_MULTIPLIERS = (2, 5, 10)  # n_init = multiplier * dim


def raw_path(problem, design, n_init, seed):
    return RAW / problem / f"{design}_n{n_init:03d}_s{seed:02d}.npz"


def _task(args):
    pname, design, n_init, seed = args
    out = raw_path(pname, design, n_init, seed)
    if out.exists():
        return
    problem = PROBLEMS[pname]()
    t0 = time.time()
    # Distinct, reproducible seed per (problem, n_init, seed), shared across designs
    # so the BO loop's own randomness is matched between designs.
    f, feas = run_bo(problem, design, n_init, problem.budget,
                     seed=1000 * seed + n_init)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, f=f, feasible=feas, n_init=n_init, seed=seed)
    print(f"{pname:16s} {design:7s} n0={n_init:3d} seed={seed:2d} "
          f"{time.time() - t0:6.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+", default=list(PROBLEMS))
    ap.add_argument("--designs", nargs="+", default=list(DESIGNS))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()

    tasks = []
    for pname in a.problems:
        d = PROBLEMS[pname]().dim
        tasks += itertools.product([pname], a.designs, [m * d for m in INIT_MULTIPLIERS],
                                   range(a.seeds))
    # Slowest problems first so the pool stays busy at the end.
    tasks.sort(key=lambda t: PROBLEMS[t[0]]().n_constraints, reverse=True)
    with Pool(a.workers) as pool:
        list(pool.imap_unordered(_task, tasks))


if __name__ == "__main__":
    main()
