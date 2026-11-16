"""Clustered XYOnion: layer-based train/test splitting for non-convex data."""

__all__ = [
    "XYOnionSplit",
    "xyonion_train_test_split",
    "xyonion_train_val_test_split",
    "HDBSCAN_KNN",
]
__version__ = "0.2.0"

_SOURCES = {
    "XYOnionSplit": "xyonion",
    "xyonion_train_test_split": "xyonion",
    "xyonion_train_val_test_split": "xyonion",
    "HDBSCAN_KNN": "_clusterers",
}


def __getattr__(name):
    # Lazy import so the numpy-only core (clustered_xyonion._core) can be used
    # without importing scikit-learn.
    if name in _SOURCES:
        from importlib import import_module

        return getattr(import_module(f".{_SOURCES[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
