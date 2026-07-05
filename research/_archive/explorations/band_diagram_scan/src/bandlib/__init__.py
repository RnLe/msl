"""Bandlib – utilities to generate and store 2D photonic band-diagram libraries."""
from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - executed only when installed as a package
    __version__ = version("bandlib")
except PackageNotFoundError:  # pragma: no cover - local editable installs
    __version__ = "0.0.0"

__all__ = ["__version__"]
