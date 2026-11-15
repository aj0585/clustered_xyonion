"""Clustered XYOnion: layer-based train/test splitting for non-convex data."""

__all__ = ["XYOnion", "xyonion_train_test_split"]
__version__ = "0.1.0"


def __getattr__(name):
    # Lazy import so the numpy-only core (clustered_xyonion._core) can be used
    # without importing scikit-learn.
    if name in __all__:
        from . import xyonion

        return getattr(xyonion, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
