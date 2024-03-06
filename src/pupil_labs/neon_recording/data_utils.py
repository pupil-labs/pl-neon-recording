import pathlib
import typing

import numpy as np

from . import structlog
log = structlog.get_logger(__name__)

def load_with_error_check(func: typing.Callable, fpath: pathlib.Path, err_msg_supp: str):
    log.debug(f"NeonRecording: Loading {fpath.name}")

    try:
        res = func(fpath)
        if res is not None:
            return res
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {fpath.name}. {err_msg_supp}")
    except OSError:
        raise OSError(f"Error opening file: {fpath.name}.")
    except Exception as e:
        log.exception(f"Unexpected error loading {fpath.name}: {e}")
        raise


# obtained from @dom:
# https://github.com/pupil-labs/neon-player/blob/master/pupil_src/shared_modules/pupil_recording/update/neon.py
#
# can return none, as worn data is always 1 for Neon
def load_worn_data(path: pathlib.Path):
    log.debug("NeonRecording: Loading worn data.")

    if not (path and path.exists()):
        return None

    confidences = np.fromfile(str(path), "<u1") / 255.0
    return np.clip(confidences, 0.0, 1.0)
