"""Aggregate the skopt vs BoTorch runs into a CSV, convergence plots and a summary.

    python -m bo_bench.botorch_compare.report [--out reports/botorch_compare]
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

from ..init_study.problems import PROBLEMS  # noqa: E402
from ..init_study.report import best_so_far, regret  # noqa: E402
from ..plotting import MUTED, SERIES_COLORS, TEXT  # noqa: E402
from .methods import METHODS  # noqa: E402
from .run import RAW  # noqa: E402


def load(raw_dir, pname):
    """{method: list of (seed, f, feasible, seconds)}"""
    runs = {}
    for path in sorted((raw_dir / pname).glob("*.npz")):
        z = np.load(path)
        method = path.name.rsplit("_s", 1)[0]
        runs.setdefault(method, []).append(
            (int(z["seed"]), z["f"], z["feasible"], float(z["seconds"]), int(z["n_init"])))
    return runs


def write_csv(path, pname, runs):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "method", "seed", "iteration", "value", "feasible", "best_so_far"])
        for method, rs in sorted(runs.items()):
            for seed, f, feas, _, _ in rs:
                for i, (v, ok, b) in enumerate(zip(f, feas, best_so_far(f, feas)), start=1):
                    w.writerow([pname, method, seed, i, f"{v:.10g}", int(ok), f"{b:.10g}"])


def plot_problem(path, problem, runs, methods):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    n0 = None
    for i, method in enumerate(methods):
        rs = runs.get(method)
        if not rs:
            continue
        n0 = rs[0][4]
        R = np.array([regret(problem, best_so_far(f, feas)) for _, f, feas, _, _ in rs])
        x = np.arange(1, R.shape[1] + 1)
        with np.errstate(invalid="ignore"):
            med = np.median(R, axis=0)
            q25, q75 = np.percentile(R, [25, 75], axis=0)
        med[~np.isfinite(med)] = np.nan
        q75 = np.where(np.isfinite(q75), q75, np.nan)
        ax.fill_between(x, q25, q75, color=SERIES_COLORS[i], alpha=0.15, linewidth=0)
        ax.plot(x, med, color=SERIES_COLORS[i], linewidth=1.8, label=method)
    if n0:
        ax.axvline(n0 + 0.5, color=MUTED, linewidth=1, linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel("Evaluation", color=TEXT)
    ylabel = "Best-so-far regret" if problem.f_opt == 0 else "Best feasible, relative regret"
    ax.set_ylabel(f"{ylabel} (log)", color=TEXT)
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    ax.set_title(f"{problem.name}: median and IQR over seeds", color=TEXT, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _fmt(v):
    return "∞" if not np.isfinite(v) else f"{v:.3g}"


def _q(a):
    with np.errstate(invalid="ignore"):
        return f"{_fmt(np.median(a))} [{_fmt(np.percentile(a, 25))}, {_fmt(np.percentile(a, 75))}]"


def summarise(problem, runs, methods):
    lines = [f"### {problem.name} (d={problem.dim}, budget={problem.budget}, best known {problem.f_opt:g})", "",
             "Final best feasible objective, median [25th, 75th percentile] over seeds. "
             "`p vs skopt` is a two-sided Mann-Whitney U test on the final values. "
             "Time is median wall-clock seconds per run on one CPU thread.", "",
             "| method | seeds | final | median relative/abs. regret | p vs skopt | feasible at end | time (s) |",
             "| :--- | ---: | :--- | ---: | ---: | ---: | ---: |"]
    finals = {m: np.array([best_so_far(f, feas)[-1] for _, f, feas, _, _ in runs[m]])
              for m in methods if m in runs}
    for m, fin in finals.items():
        p = "" if m == "skopt" or "skopt" not in finals else \
            f"{mannwhitneyu(fin, finals['skopt']).pvalue:.3f}"
        secs = np.median([r[3] for r in runs[m]])
        with np.errstate(invalid="ignore"):
            reg = regret(problem, np.median(fin))
        lines.append(f"| {m} | {len(fin)} | {_q(fin)} | {_fmt(reg)} | {p} | "
                     f"{np.isfinite(fin).sum()}/{len(fin)} | {secs:.1f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=Path("reports/botorch_compare"))
    ap.add_argument("--problems", nargs="+", default=list(PROBLEMS))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    methods = list(METHODS)
    sections = []
    for pname in a.problems:
        runs = load(a.raw, pname)
        if not runs:
            continue
        problem = PROBLEMS[pname]()
        write_csv(a.raw.parent / f"{pname}_results.csv", pname, runs)
        plot_problem(a.out / f"{pname}_convergence.png", problem, runs, methods)
        sections.append(summarise(problem, runs, methods) + f"\n![{pname}]({pname}_convergence.png)\n")
    (a.out / "summary.md").write_text("# skopt vs BoTorch: results\n\n" + "\n".join(sections))
    print(f"Wrote {a.out / 'summary.md'}")


if __name__ == "__main__":
    main()
