from pathlib import Path
from typing import Tuple

import numpy as np


def find_sorted_multipart_files(
    recording_path: Path, basename: str, extension: str = ".raw"
):
    file_pairs = []
    for raw_file in recording_path.glob(f"{basename} ps*{extension}"):
        time_file = raw_file.with_suffix(".time")
        if time_file.exists():
            file_pairs.append((raw_file, time_file))

    return sorted(file_pairs, key=lambda pair: int(pair[0].stem[len(basename) + 3 :]))


def load_multipart_data_time_pairs(
    file_pairs, dtype, field_count
) -> Tuple[np.ndarray, np.ndarray]:
    data_buffer = b""
    ts_buffer = b""
    for data_file, time_file in file_pairs:
        with open(data_file, "rb") as f:
            data_buffer += f.read()
        with open(time_file, "rb") as f:
            ts_buffer += f.read()

    if dtype == "str":
        data = np.array(data_buffer.decode().rstrip("\n").split("\n"))
    else:
        data = np.frombuffer(data_buffer, dtype).reshape([-1, field_count])

    timestamps = np.frombuffer(ts_buffer, dtype="<u8").astype(np.float64) * 1e-9

    return data, timestamps


def load_and_convert_tstamps(path: Path):
    return np.fromfile(str(path), dtype="<u8").astype(np.float64) * 1e-9


def load_multipart_timestamps(files):
    ts_buffer = b""
    for time_file in files:
        with open(time_file, "rb") as f:
            ts_buffer += f.read()

    timestamps = np.frombuffer(ts_buffer, dtype="<u8").astype(np.float64) * 1e-9

    return timestamps
