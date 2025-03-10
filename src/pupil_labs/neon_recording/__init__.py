import importlib.metadata
import logging
import os

from .neon_recording import load

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"


log = logging.getLogger(__name__)

level = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, level)


__all__ = ["__version__", "load"]
