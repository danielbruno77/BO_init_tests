"""Convergence plots."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Fixed categorical order, assigned by method position (never cycled).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT = "#0b0b0b"
MUTED = "#52514e"


def plot_convergence(curves, path, title="", n_init=None, optimum=None):
    """curves: {method: array of shape (n_seeds, budget)} of best-so-far values."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ends = []
    for i, (method, runs) in enumerate(curves.items()):
        color = SERIES_COLORS[i]
        x = np.arange(1, runs.shape[1] + 1)
        median = np.median(runs, axis=0)
        q25, q75 = np.percentile(runs, [25, 75], axis=0)
        ax.fill_between(x, q25, q75, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, median, color=color, linewidth=2, label=f"{method} (median, IQR)")
        ends.append((median[-1], x[-1], method))

    if optimum is not None:
        ax.axhline(optimum, color=MUTED, linewidth=1, linestyle=":")
        ax.annotate("global optimum", (0.5, optimum), xycoords=("axes fraction", "data"),
                    xytext=(0, -10), textcoords="offset points", ha="center",
                    color=MUTED, fontsize=8)
    if n_init:
        ax.axvline(n_init + 0.5, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate("end of random init", (n_init + 0.5, 0), xycoords=("data", "axes fraction"),
                    xytext=(4, 6), textcoords="offset points", color=MUTED, fontsize=8)

    _label_line_ends(fig, ax, ends)

    ax.set_xlabel("Evaluation", color=TEXT)
    ax.set_ylabel("Best objective so far (higher is better)", color=TEXT)
    ax.set_title(title, color=TEXT, loc="left")
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="lower right")
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _label_line_ends(fig, ax, ends, min_gap_pt=12):
    """Direct-label each line at its right end, nudging labels apart so they don't overlap."""
    fig.canvas.draw()
    min_gap_px = min_gap_pt * fig.dpi / 72
    to_px = ax.transData.transform
    placed = []
    for y, x, method in sorted(ends):
        px_y = to_px((x, y))[1]
        if placed and px_y - placed[-1] < min_gap_px:
            px_y = placed[-1] + min_gap_px
        placed.append(px_y)
        offset = px_y - to_px((x, y))[1]
        ax.annotate(f"{method}  {y:.3g}", (x, y), xytext=(6, offset * 72 / fig.dpi),
                    textcoords="offset points", va="center", color=TEXT, fontsize=9)
