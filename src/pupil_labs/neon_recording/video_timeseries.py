from logging import getLogger
from pathlib import Path

from pupil_labs.video import MultiPartReader

from .utils import find_sorted_multipart_files, load_multipart_timestamps

log = getLogger(__name__)


class VideoTimeseries(MultiPartReader):
    def __init__(self, rec_dir: Path, base_name: str):
        log.debug(f"NeonRecording: Loading video: {base_name}.")
        video_files = find_sorted_multipart_files(rec_dir, base_name, ".mp4")
        self.timestamps = load_multipart_timestamps([p[1] for p in video_files])
        super().__init__([p[0] for p in video_files])
