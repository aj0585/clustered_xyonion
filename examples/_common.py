"""Shared helpers for the examples: fake data sets, a pretend HDBSCAN_KNN and
plotting.

Every example saves its figures to ``examples/output/``. Add ``--show`` to the
command line to also open them in a window.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

if "--show" not in sys.argv:
    matplotlib.use("Agg")  # draw to files only; no window needed

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from clustered_xyonion import HDBSCAN_KNN  # noqa: E402
from clustered_xyonion._core import TEST, TRAIN, VALIDATION  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# --- palette (light mode) -----------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots in fixed order (never cycled), plus a marker per slot so
# cluster identity never relies on colour alone.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# Train is the majority, so it recedes; the two held-out sets carry the colour.
SUBSET_STYLE = {
    TRAIN: {"name": "train", "color": BASELINE, "size": 14},
    VALIDATION: {"name": "validation", "color": "#eb6834", "size": 26},
    TEST: {"name": "test", "color": "#2a78d6", "size": 26},
}

# One-hue sequential ramps, light -> dark.
RESPONSE_CMAP = LinearSegmentedColormap.from_list(
    "response", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DEPTH_CMAP = LinearSegmentedColormap.from_list(
    "depth", ["#fbd9c9", "#f5b496", "#f08d63", "#eb6834", "#c4521f", "#8f3a16"])


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "font.family": "sans-serif",
        "font.size": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


apply_style()


# --- pretend HDBSCAN_KNN ------------------------------------------------


class PretendHDBSCAN_KNN(HDBSCAN_KNN):
    """Stands in for HDBSCAN_KNN, which isn't implemented yet.

    It takes the same parameters (so XYOnion can pass ``min_cluster_size``,
    ``n_neighbors`` etc. down to it as usual), but instead of clustering it
    just returns the made-up ``fake_labels`` it was given.
    """

    def __init__(self, fake_labels=None, min_cluster_size=5, min_samples=None,
                 cluster_selection_epsilon=0.0, metric="euclidean", n_neighbors=5):
        self.fake_labels = fake_labels
        super().__init__(min_cluster_size=min_cluster_size, min_samples=min_samples,
                         cluster_selection_epsilon=cluster_selection_epsilon,
                         metric=metric, n_neighbors=n_neighbors)

    def fit(self, X, y=None):
        self.labels_ = np.asarray(self.fake_labels)
        if self.labels_.shape[0] != np.asarray(X).shape[0]:
            raise ValueError("fake_labels must have one label per sample.")
        return self


# --- fake data sets -----------------------------------------------------


def make_strings_2d(n_per=250, n_strings=3, spacing=5.0, noise=0.15, seed=0):
    """Long, wavy, string-like clusters stacked above each other.

    The response varies along each string and differs between strings.
    Returns X (n, 2), y (n,), true string label (n,).
    """
    rng = np.random.RandomState(seed)
    Xs, ys, labels = [], [], []
    for k in range(n_strings):
        t = rng.uniform(0, 1, n_per)
        x0 = 12 * t
        x1 = 1.2 * np.sin(2.5 * np.pi * t + k) + spacing * k
        Xs.append(np.column_stack([x0, x1]) + noise * rng.normal(size=(n_per, 2)))
        ys.append(5 * t + 2 * k + 0.6 * np.sin(6 * np.pi * t) + 0.2 * rng.normal(size=n_per))
        labels.append(np.full(n_per, k))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(labels)


def make_helices_3d(n_per=300, n_helices=3, spacing=3.5, noise=0.08, seed=0):
    """Intertwined-looking helices standing side by side in 3D.

    Returns X (n, 3), y (n,), true helix label (n,).
    """
    rng = np.random.RandomState(seed)
    Xs, ys, labels = [], [], []
    for k in range(n_helices):
        t = rng.uniform(0, 4 * np.pi, n_per)
        pts = np.column_stack([spacing * k + np.cos(t), np.sin(t), t / 2])
        Xs.append(pts + noise * rng.normal(size=pts.shape))
        ys.append(t / 2 + 1.5 * k + 0.5 * np.cos(t) + 0.15 * rng.normal(size=n_per))
        labels.append(np.full(n_per, k))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(labels)


# --- plotting -----------------------------------------------------------


def _scatter(ax, X, **kwargs):
    if X.shape[1] == 3:
        return ax.scatter(X[:, 0], X[:, 1], X[:, 2], **kwargs)
    return ax.scatter(X[:, 0], X[:, 1], **kwargs)


def _make_axes(fig, spec, X):
    """2D or 3D axes (matching X's columns) with equal scaling on every axis,
    so shapes aren't stretched."""
    if X.shape[1] == 3:
        ax = fig.add_subplot(spec, projection="3d")
        ax.set_facecolor(SURFACE)
        ax.set_box_aspect(np.ptp(X, axis=0))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            # At most 3 ticks per axis so a short axis's labels don't collide.
            axis.set_major_locator(MaxNLocator(3))
            axis.pane.set_facecolor(SURFACE)
            axis.pane.set_edgecolor(GRIDLINE)
        return ax
    ax = fig.add_subplot(spec)
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def _legend_below(ax, handles=None):
    """Legend in one row under the plot, so it never covers data."""
    kwargs = {} if handles is None else {"handles": handles}
    n = len(handles) if handles is not None else len(ax.get_legend_handles_labels()[0])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=7,
              ncol=min(n, 4), handletextpad=0.3, columnspacing=1.0, **kwargs)


def plot_response(ax, X, y):
    """Colour each sample by its (fake) response value."""
    points = _scatter(ax, X, c=y, cmap=RESPONSE_CMAP, s=14, linewidths=0)
    ax.set_title("Response y")
    plt.colorbar(points, ax=ax, shrink=0.75, label="y")


def plot_clusters(ax, X, labels):
    """Colour and shape each sample by its cluster; label each cluster centre."""
    handles = []
    clusters = [g for g in np.unique(labels) if g != -1]
    if len(clusters) > len(CATEGORICAL):
        raise ValueError("More clusters than colour slots; use fewer clusters.")
    for g in np.unique(labels):
        members = labels == g
        if g == -1:
            color, marker, name = MUTED, ".", "noise"
        else:
            slot = clusters.index(g)
            color, marker, name = CATEGORICAL[slot], MARKERS[slot], f"C{g}"
        _scatter(ax, X[members], color=color, marker=marker, s=14,
                 edgecolors=SURFACE, linewidths=0.4)
        handles.append(Line2D([], [], color=color, marker=marker, linestyle="", label=name))
        if g != -1 and X.shape[1] == 2:
            cx, cy = X[members].mean(axis=0)
            ax.annotate(name, (cx, cy), color=INK, fontsize=8, fontweight="bold",
                        ha="center", va="center",
                        bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.8})
    ax.set_title(f"Clusters ({len(handles)})")
    _legend_below(ax, handles)


def layer_depth(info):
    """Layer index scaled to 0 (outermost) .. 1 (innermost) within each
    cluster; NaN for samples placed by the final random split."""
    depth = np.full(info.layer.shape, np.nan)
    for g in np.unique(info.cluster_labels):
        members = (info.cluster_labels == g) & (info.layer >= 0)
        if members.any():
            deepest = max(info.layer[members].max(), 1)
            depth[members] = info.layer[members] / deepest
    return depth


def plot_layers(ax, X, info, colorbar=True):
    """Colour each sample by how deep its onion layer is within its cluster."""
    depth = layer_depth(info)
    core = np.isnan(depth)
    # Most samples are usually NOT layered (with the defaults only ~27% are);
    # show them as small grey dots so the layers stand out.
    _scatter(ax, X[core], color=BASELINE, s=6, linewidths=0, label="random split (not layered)")
    points = _scatter(ax, X[~core], c=depth[~core], cmap=DEPTH_CMAP, vmin=0, vmax=1,
                      s=14, linewidths=0)
    n_layers = len(np.unique(info.layer[info.layer >= 0]))
    ax.set_title(f"Onion layers (up to {n_layers} per cluster)")
    if colorbar:
        add_depth_colorbar(points, ax)
    if core.any():
        _legend_below(ax)
    return points


def add_depth_colorbar(points, ax):
    bar = plt.colorbar(points, ax=ax, shrink=0.75, ticks=[0, 1])
    bar.ax.set_yticklabels(["outer", "inner"])


def plot_subsets(ax, X, info, title="Split"):
    """Colour each sample by the subset XYOnion put it in."""
    handles = []
    for code in (TRAIN, VALIDATION, TEST):
        members = info.subset == code
        if not members.any():
            continue
        style = SUBSET_STYLE[code]
        _scatter(ax, X[members], color=style["color"], s=style["size"],
                 edgecolors=SURFACE, linewidths=0.5)
        handles.append(Line2D([], [], color=style["color"], marker="o", linestyle="",
                              label=f"{style['name']} ({members.sum()})"))
    ax.set_title(title)
    _legend_below(ax, handles)


def plot_response_coverage(ax, y, info):
    """One row per subset: every response value as a dot, with its range.

    Shows whether validation and test responses sit inside the train range
    (no extrapolation) while still spanning most of it. Dashed lines mark the
    train range; any held-out value outside it is counted in the row's label.
    """
    rng = np.random.RandomState(0)
    codes = [c for c in (TRAIN, VALIDATION, TEST) if (info.subset == c).any()]
    train_y = y[info.subset == TRAIN]
    lo, hi = train_y.min(), train_y.max()
    for edge in (lo, hi):
        ax.axvline(edge, color=MUTED, linewidth=1, linestyle="--", zorder=0)
    for row, code in enumerate(codes):
        values = y[info.subset == code]
        style = SUBSET_STYLE[code]
        jitter = row + 0.25 * (rng.uniform(size=values.size) - 0.5)
        color = style["color"] if code != TRAIN else MUTED
        ax.scatter(values, jitter, color=color, s=10, linewidths=0, alpha=0.8)
        ax.plot([values.min(), values.max()], [row + 0.3] * 2, color=INK_SECONDARY, linewidth=2)
        label = f"  {values.min():.2f} to {values.max():.2f}"
        if code != TRAIN:
            outside = int(((values < lo) | (values > hi)).sum())
            label += f"  ({outside} outside train range)"
        ax.text(values.max(), row + 0.3, label, va="center", fontsize=7, color=INK_SECONDARY)
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels([SUBSET_STYLE[c]["name"] for c in codes])
    ax.set_xlabel("y")
    ax.set_title("Response coverage by subset")
    ax.grid(axis="y", visible=False)


def plot_split(X, y, info, title, subtitle, filename):
    """Five-panel overview of one XYOnion split, saved to examples/output/."""
    fig = plt.figure(figsize=(15, 10.5))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.45, wspace=0.25)
    plot_response(_make_axes(fig, grid[0, 0], X), X, y)
    plot_clusters(_make_axes(fig, grid[0, 1], X), X, info.cluster_labels)
    plot_layers(_make_axes(fig, grid[0, 2], X), X, info)
    plot_subsets(_make_axes(fig, grid[1, 0], X), X, info)
    plot_response_coverage(fig.add_subplot(grid[1, 1:]), y, info)
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.05, ha="left")
    fig.text(0.05, 0.935, subtitle, fontsize=9, color=INK_SECONDARY, ha="left")
    save(fig, filename)


def save(fig, filename):
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"saved {path}")
    if "--show" in sys.argv:
        plt.show()
    plt.close(fig)


def describe(cv) -> str:
    """One-line summary of a splitter's settings, for figure subtitles."""
    clustering = cv.clustering if isinstance(cv.clustering, (str, type(None))) \
        else type(cv.clustering).__name__
    parts = [f"test_fraction={cv.test_fraction}"]
    if cv.validation_fraction:
        parts.append(f"validation_fraction={cv.validation_fraction}")
    parts += [f"n_onion={cv.n_onion}", f"loop_fraction={cv.loop_fraction}", f"metric={cv.metric}",
              f"clustering={clustering}"]
    if cv.clustering_params:
        parts.append(f"clustering_params={cv.clustering_params}")
    return ",  ".join(parts)
