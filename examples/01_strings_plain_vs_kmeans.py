"""Why cluster first: plain XYOnion vs kmeans + XYOnion on string-like data.

Three long, wavy strings. Plain XYOnion treats the whole data set as one
cloud, so its onion layers cut across the strings: the string ends and the
outer strings are peeled first and the middle string ends up deep inside.
With kmeans first (enough clusters to cut each string into roughly convex
pieces), every piece gets its own onion, so train, validation and test are
spread along every string.

Run:  uv run --group examples python examples/01_strings_plain_vs_kmeans.py
"""

from _common import describe, make_strings_2d, plot_split

from clustered_xyonion import XYOnionSplit

X, y, _ = make_strings_2d()

for name, cv in [
    ("plain", XYOnionSplit(test_fraction=0.2, validation_fraction=0.2, random_state=0)),
    ("kmeans", XYOnionSplit(test_fraction=0.2, validation_fraction=0.2,
                            clustering="kmeans", clustering_params={"n_clusters": 8},
                            random_state=0)),
]:
    info = cv.split_info(X, y)
    title = ("Plain XYOnion on string-like clusters" if name == "plain"
             else "kmeans + XYOnion on string-like clusters")
    plot_split(X, y, info, title, describe(cv), f"01_strings_{name}.png")
