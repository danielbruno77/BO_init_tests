"""Aggregate the raw runs into a CSV, convergence plots and a markdown summary.

    python -m bo_bench.init_study.report [--raw results/init_study/raw] [--out reports/init_study]
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

from ..plotting import MUTED, SERIES_COLORS, TEXT  # noqa: E402
from .designs import DESIGNS  # noqa: E402
from .problems import PROBLEMS  # noqa: E402
from .run import RAW  # noqa: E402


def best_so_far(f, feasible):
    """Best feasible value after each evaluation (inf until the first feasible point)."""
    return np.minimum.accumulate(np.where(feasible, f, np.inf))


def regret(problem, best):
    """Absolute regret for Rastrigin (optimum 0), relative regret otherwise."""
    if problem.f_opt == 0:
        return best - problem.f_opt
    return (best - problem.f_opt) / abs(problem.f_opt)


def load(raw_dir, pname):
    """{(design, n_init): list of (seed, f, feasible)}"""
    runs = {}
    for path in sorted((raw_dir / pname).glob("*.npz")):
        z = np.load(path)
        design = path.name.split("_n")[0]
        runs.setdefault((design, int(z["n_init"])), []).append((int(z["seed"]), z["f"], z["feasible"]))
    return runs


def write_csv(path, pname, runs):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "design", "n_init", "seed", "iteration", "value", "feasible", "best_so_far"])
        for (design, n0), rs in sorted(runs.items()):
            for seed, f, feas in rs:
                for i, (v, ok, b) in enumerate(zip(f, feas, best_so_far(f, feas)), start=1):
                    w.writerow([pname, design, n0, seed, i, f"{v:.10g}", int(ok), f"{b:.10g}"])


def plot_problem(path, problem, runs, n_inits, designs):
    fig, axes = plt.subplots(1, len(n_inits), figsize=(4.2 * len(n_inits), 4.0), dpi=150,
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, n0 in zip(axes[0], n_inits):
        ax.set_facecolor("#fcfcfb")
        for i, design in enumerate(designs):
            rs = runs.get((design, n0))
            if not rs:
                continue
            R = np.array([regret(problem, best_so_far(f, feas)) for _, f, feas in rs])
            x = np.arange(1, R.shape[1] + 1)
            with np.errstate(invalid="ignore"):
                med = np.median(R, axis=0)
                q25, q75 = np.percentile(R, [25, 75], axis=0)
            med[~np.isfinite(med)] = np.nan
            q75 = np.where(np.isfinite(q75), q75, np.nan)
            ax.fill_between(x, q25, q75, color=SERIES_COLORS[i], alpha=0.15, linewidth=0)
            ax.plot(x, med, color=SERIES_COLORS[i], linewidth=1.8, label=design)
        ax.axvline(n0 + 0.5, color=MUTED, linewidth=1, linestyle="--")
        ax.set_yscale("log")
        ax.set_title(f"n_init = {n0}", color=TEXT, loc="left", fontsize=10)
        ax.set_xlabel("Evaluation", color=TEXT)
        ax.grid(True, color="#e6e5e0", linewidth=0.8, which="major")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED)
    ylabel = "Best-so-far regret" if problem.f_opt == 0 else "Best feasible, relative regret"
    axes[0, 0].set_ylabel(f"{ylabel} (log)", color=TEXT)
    axes[0, -1].legend(frameon=False, loc="upper right", fontsize=8)
    fig.suptitle(f"{problem.name}: median and IQR over seeds (dashed line = end of initial design)",
                 x=0.01, ha="left", color=TEXT, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _fmt(v):
    return "∞" if not np.isfinite(v) else f"{v:.3g}"


def summarise(problem, runs, n_inits, designs):
    lines = [f"### {problem.name} (d={problem.dim}, budget={problem.budget}, best known {problem.f_opt:g})", "",
             "Final best feasible objective; median [25th, 75th percentile] over seeds. "
             "`p vs random` is a two-sided Mann-Whitney U test on the final values.", "",
             "| n_init | design | seeds | after initial design | final | p vs random | runs feasible at end |",
             "| ---: | :--- | ---: | :--- | :--- | ---: | ---: |"]
    for n0 in n_inits:
        finals = {}
        for design in designs:
            rs = runs.get((design, n0))
            if rs:
                finals[design] = np.array([best_so_far(f, feas)[-1] for _, f, feas in rs])
        for design, fin in finals.items():
            rs = runs[(design, n0)]
            init = np.array([best_so_far(f, feas)[n0 - 1] for _, f, feas in rs])
            q = lambda a: f"{_fmt(np.median(a))} [{_fmt(np.percentile(a, 25))}, {_fmt(np.percentile(a, 75))}]"
            p = "" if design == "random" or "random" not in finals else \
                f"{mannwhitneyu(fin, finals['random']).pvalue:.2f}"
            lines.append(f"| {n0} | {design} | {len(fin)} | {q(init)} | {q(fin)} | {p} | "
                         f"{np.isfinite(fin).sum()}/{len(fin)} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=Path("reports/init_study"))
    ap.add_argument("--problems", nargs="+", default=list(PROBLEMS))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    designs = list(DESIGNS)
    sections = []
    for pname in a.problems:
        runs = load(a.raw, pname)
        if not runs:
            continue
        problem = PROBLEMS[pname]()
        n_inits = sorted({n0 for _, n0 in runs})
        write_csv(a.raw.parent / f"{pname}_results.csv", pname, runs)
        plot_problem(a.out / f"{pname}_convergence.png", problem, runs, n_inits, designs)
        sections.append(summarise(problem, runs, n_inits, designs)
                        + f"\n![{pname}]({pname}_convergence.png)\n")
    (a.out / "summary.md").write_text("# Initial-design comparison: results\n\n" + "\n".join(sections))
    print(f"Wrote {a.out / 'summary.md'}")


if __name__ == "__main__":
    main()
