"""Top-level entry-point for the pl-neon-recording package"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("pupil_labs.neon_recording")
except PackageNotFoundError:
    # package is not installed
    pass

from .neon_recording import load

__all__ = [
    "__version__",
    "load"
    ]
