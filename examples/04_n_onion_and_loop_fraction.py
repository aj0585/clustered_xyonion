"""How n_onion and loop_fraction change the onion layers.

Each of the n_onion rounds peels one layer per subset (train, then test, then
validation). A subset's layer holds round(loop_fraction * m0 * fraction)
samples, where m0 is the number of samples not yet assigned at the start of
the round, so the layers shrink round by round. Everything left after the
last round is split at random.

So n_onion and loop_fraction together decide how much of the data is placed
by the onion (spread-out, outside-first) and how much at random:
the defaults (3 rounds of 10%) layer about 27%.

Run:  uv run --group examples python examples/04_n_onion_and_loop_fraction.py
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import INK_SECONDARY, add_depth_colorbar, plot_layers, plot_subsets, save

from clustered_xyonion import XYOnionSplit

rng = np.random.RandomState(0)
X = rng.normal(size=(600, 2)) @ np.array([[1.0, 0.3], [0.0, 0.6]])
y = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.2 * rng.normal(size=len(X))

settings = [
    {"n_onion": 3, "loop_fraction": 0.1},   # the reference defaults
    {"n_onion": 3, "loop_fraction": 0.3},
    {"n_onion": 10, "loop_fraction": 0.2},
]
# Shared axes so all six panels use exactly the same scale.
fig, axes = plt.subplots(2, len(settings), figsize=(14, 8),
                         sharex=True, sharey=True, constrained_layout=True)
for col, kw in enumerate(settings):
    cv = XYOnionSplit(test_fraction=0.2, validation_fraction=0.2,
                      metric="mahalanobis", random_state=0, **kw)
    info = cv.split_info(X, y)
    layered = (info.layer >= 0).mean()
    name = f"n_onion={kw['n_onion']}, loop_fraction={kw['loop_fraction']}"
    points = plot_layers(axes[0, col], X, info, colorbar=False)
    axes[0, col].set_title(f"{name}\n{info.layer.max() + 1} layers, {layered:.0%} layered")
    plot_subsets(axes[1, col], X, info, title=f"{name}: split")
for ax in axes.flat:
    ax.set_aspect("equal")
add_depth_colorbar(points, axes[0, :].tolist())

fig.suptitle("Effect of n_onion and loop_fraction", fontsize=14, fontweight="bold",
             x=0.02, ha="left")
fig.text(0.02, 0.955, "Single tilted cloud, 600 samples, test_fraction=0.2, "
         "validation_fraction=0.2, metric=mahalanobis, no clustering",
         fontsize=9, color=INK_SECONDARY, ha="left")
save(fig, "04_n_onion_and_loop_fraction.png")
