import pathlib

import pupil_labs.video as plv

from ... import structlog
from ...time_utils import load_and_convert_tstamps

log = structlog.get_logger(__name__)


def _load_av_container(rec_dir: pathlib.Path, file_name: pathlib.Path | str):
    log.debug(
        f"NeonRecording: Loading video and associated timestamps: {file_name}."
    )

    if not (rec_dir / (file_name + ".mp4")).exists():
        raise FileNotFoundError(
            f"File not found: {rec_dir / (file_name + '.mp4')}. Please double check the recording download."
        )

    container = plv.open(rec_dir / (file_name + ".mp4"))

    # use hardware ts
    # ts = load_and_convert_tstamps(rec_dir / (file_name + '.time_aux'))
    try:
        video_ts = load_and_convert_tstamps(
            rec_dir / (file_name + ".time")
        )
    except Exception as e:
        log.exception(f"Error loading timestamps: {e}")
        raise

    return container, video_ts
