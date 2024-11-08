from pathlib import Path

import numpy as np
import numpy.typing as npt

from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_data_time_pairs,
)


class Gaze(dict):
    timestamps: npt.NDArray[np.int64]
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]

    def __init__(self, rec_dir: Path):
        gaze_200hz_file = rec_dir / "gaze_200hz.raw"
        time_200hz_file = rec_dir / "gaze_200hz.time"
        gaze_file_pairs = []
        if gaze_200hz_file.exists() and time_200hz_file.exists():
            gaze_file_pairs.append((gaze_200hz_file, time_200hz_file))
        else:
            gaze_file_pairs = find_sorted_multipart_files(rec_dir, "gaze")
        gaze_data, time_data = load_multipart_data_time_pairs(gaze_file_pairs, "<f4", 2)

        self["timestamps"] = time_data
        self["x"] = gaze_data[:, 0]
        self["y"] = gaze_data[:, 1]

    @property
    def xy(self) -> npt.NDArray[np.float64]:
        return np.column_stack([self.x, self.y])

    def __getattr__(self, key: str) -> npt.NDArray:
        return self[key]


if __name__ == "__main__":
    path = Path("/home/marc/pupil_labs/pl-matching/tests/data/demo_recording")
    gaze = Gaze(path)
    pass
