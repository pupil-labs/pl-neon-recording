from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pupil_labs.neon_recording.constants import TIMESTAMP_DTYPE
from pupil_labs.neon_recording.stream.array_record import Array


def find_sorted_multipart_files(
    recording_path: Path, basename: str, extension: str = ".raw"
):
    file_pairs = []
    for raw_file in recording_path.glob(f"{basename} ps*{extension}"):
        if basename == "worn":
            # The `worn` sensor doesn't have its own .time file
            gaze_time_stem = raw_file.stem.replace("worn ", "gaze ")
            time_file = raw_file.parent / f"{gaze_time_stem}.time"

        else:
            time_file = raw_file.with_suffix(".time")

        if time_file.exists():
            file_pairs.append((raw_file, time_file))

    return sorted(file_pairs, key=lambda pair: int(pair[0].stem[len(basename) + 3 :]))


def load_multipart_data_time_pairs(file_pairs, dtype):
    ts_files = [time_file for _, time_file in file_pairs]
    data_files = [data_file for data_file, _ in file_pairs]

    time_data = Array(ts_files, TIMESTAMP_DTYPE)
    if not len(time_data):
        return np.array([], dtype=TIMESTAMP_DTYPE)

    if dtype == "str":
        data_bytes = b""
        for data_file in data_files:
            with open(data_file, "rb") as f:
                data_bytes += f.read()

        item_data = np.array(data_bytes.decode().splitlines())
        item_data = item_data.view([("text", item_data.dtype)])
    else:
        item_data = Array(data_files, fallback_dtype=dtype)

    merged = join_struct_arrays([time_data, item_data])
    return merged


def load_and_convert_tstamps(path: Path):
    return np.fromfile(str(path), dtype="<i8")


def load_multipart_timestamps(files):
    ts_buffer = b""
    for time_file in files:
        with open(time_file, "rb") as f:
            ts_buffer += f.read()

    timestamps = np.frombuffer(ts_buffer, dtype="<i8")

    return timestamps


def join_struct_arrays(arrays: Sequence[npt.NDArray]):
    newdtype = [desc for a in arrays for desc in a.dtype.descr]
    newrecarray = np.empty(len(arrays[0]), dtype=newdtype)
    for a in arrays:
        assert a.dtype.names
        for name in a.dtype.names:
            newrecarray[name] = a[name]
    return newrecarray
