import pathlib

import numpy as np

from . import structlog

log = structlog.get_logger(__name__)


def ns_to_s(ns):
    SECONDS_PER_NANOSECOND = 1e-9
    return ns * SECONDS_PER_NANOSECOND


# adapted from @dom and @pfaion:
# https://github.com/pupil-labs/neon-player/blob/master/pupil_src/shared_modules/pupil_recording/update/update_utils.py
#
# modified according to mpk's note
def load_and_convert_tstamps(path: pathlib.Path):
    log.debug(f"NeonRecording: Loading timestamps from: {path.name}")

    out = np.fromfile(str(path), dtype="<u8").astype(np.float64)

    return ns_to_s(out)
