"""kmeans + XYOnion on 3D helices (train/validation/test).

Three helices standing side by side. kmeans cuts each helix into a few
pieces, and XYOnion splits each piece on its own, so all three subsets run the
full height of every helix.

Run:  uv run --group examples python examples/05_helices_3d_kmeans.py
      (add --show to rotate the 3D plots in a window)
"""

from _common import describe, make_helices_3d, plot_split

from clustered_xyonion import XYOnionSplit

X, y, _ = make_helices_3d()

cv = XYOnionSplit(
    test_fraction=0.15,
    validation_fraction=0.15,
    loop_fraction=0.2,
    clustering="kmeans",
    clustering_params={"n_clusters": 6},
    random_state=0,
)
info = cv.split_info(X, y)
plot_split(X, y, info, "3D helices: kmeans + XYOnion", describe(cv), "05_helices_3d_kmeans.png")
