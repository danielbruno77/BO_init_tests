"""Run skopt vs BoTorch (and the in-house sklearn loop) on the init-study problems.

Every method gets the same Latin-hypercube initial design of 2d points for a
given (problem, seed), and the same total budget as the initial-design study.
Each run is stored as its own .npz under results/botorch_compare/raw/, so an
interrupted sweep resumes where it stopped.

    python -m bo_bench.botorch_compare.run                         # full sweep
    python -m bo_bench.botorch_compare.run --problems rastrigin2d --seeds 2
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import itertools  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from ..init_study.designs import initial_design  # noqa: E402
from ..init_study.problems import PROBLEMS  # noqa: E402
from .methods import DEFAULT_METHODS, METHODS  # noqa: E402

RAW = Path(__file__).resolve().parents[2] / "results" / "botorch_compare" / "raw"
INIT_MULTIPLIER = 2  # best setting in the initial-design study


def raw_path(problem, method, seed):
    return RAW / problem / f"{method}_s{seed:02d}.npz"


def _task(args):
    pname, method, seed = args
    out = raw_path(pname, method, seed)
    if out.exists():
        return
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    problem = PROBLEMS[pname]()
    X0 = initial_design("lhs", INIT_MULTIPLIER * problem.dim, problem.dim, seed)
    f, feas, secs = METHODS[method](problem, X0, problem.budget, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, f=f, feasible=feas, seconds=secs, n_init=len(X0), seed=seed)
    print(f"{pname:16s} {method:13s} seed={seed:2d} {secs:6.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+", default=list(PROBLEMS))
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=list(METHODS))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    tasks = list(itertools.product(a.problems, a.methods, range(a.seeds)))
    tasks.sort(key=lambda t: PROBLEMS[t[0]]().n_constraints, reverse=True)
    with Pool(a.workers) as pool:
        list(pool.imap_unordered(_task, tasks))


if __name__ == "__main__":
    main()
