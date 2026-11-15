"""How loop_fraction changes the onion layers.

loop_fraction sets the layer size: each layer holds
max(1, round(loop_fraction * f * n)) samples, where f is the smaller held-out
fraction. Small values make many thin layers (fine-grained interleaving of
train, validation and test from the outside in); loop_fraction=1 makes each
held-out set one single, thick layer.

Run:  uv run --group examples python examples/04_loop_fraction.py
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import INK_SECONDARY, add_depth_colorbar, plot_layers, plot_subsets, save

from clustered_xyonion import XYOnionSplit

rng = np.random.RandomState(0)
X = rng.normal(size=(600, 2)) @ np.array([[1.0, 0.3], [0.0, 0.6]])
y = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.2 * rng.normal(size=len(X))

loop_fractions = [0.05, 0.2, 1.0]
# Shared axes so all six panels use exactly the same scale.
fig, axes = plt.subplots(2, len(loop_fractions), figsize=(14, 8),
                         sharex=True, sharey=True, constrained_layout=True)
for col, lf in enumerate(loop_fractions):
    cv = XYOnionSplit(test_fraction=0.2, validation_fraction=0.2, loop_fraction=lf,
                      metric="mahalanobis", random_state=0)
    info = cv.split_info(X, y)
    layer_size = max(1, round(lf * 0.2 * len(X)))
    n_layers = len(np.unique(info.layer[info.layer >= 0]))
    points = plot_layers(axes[0, col], X, info, colorbar=False)
    axes[0, col].set_title(f"loop_fraction={lf}: {n_layers} layers of {layer_size}")
    plot_subsets(axes[1, col], X, info, title=f"loop_fraction={lf}: split")
for ax in axes.flat:
    ax.set_aspect("equal")
add_depth_colorbar(points, axes[0, :].tolist())

fig.suptitle("Effect of loop_fraction", fontsize=14, fontweight="bold", x=0.02, ha="left")
fig.text(0.02, 0.955, "Single tilted cloud, 600 samples, test_fraction=0.2, "
         "validation_fraction=0.2, metric=mahalanobis, no clustering",
         fontsize=9, color=INK_SECONDARY, ha="left")
save(fig, "04_loop_fraction.png")
