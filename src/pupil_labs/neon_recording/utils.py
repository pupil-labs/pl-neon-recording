from functools import cached_property
from pathlib import Path
from typing import Literal, Sequence, Tuple, overload

import numpy as np
import numpy.typing as npt


def find_sorted_multipart_files(
    recording_path: Path, basename: str, extension: str = ".raw"
) -> list[Tuple[Path, Path]]:
    file_pairs = []
    for raw_file in recording_path.glob(f"{basename} ps*{extension}"):
        time_file = raw_file.with_suffix(".time")
        if time_file.exists():
            file_pairs.append((raw_file, time_file))

    return sorted(file_pairs, key=lambda pair: int(pair[0].stem[len(basename) + 3 :]))


def load_multipart_data_time_pairs(
    file_pairs: list[Tuple[Path, Path]], dtype: str, field_count: int
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

    timestamps = np.frombuffer(ts_buffer, dtype="<u8").astype(np.int64)

    return data, timestamps


@overload
def load_multipart_timestamps(
    files: Sequence[Path], concatenate: Literal[False]
) -> list[npt.NDArray[np.int64]]: ...
@overload
def load_multipart_timestamps(
    files: Sequence[Path], concatenate: Literal[True]
) -> npt.NDArray[np.int64]: ...
def load_multipart_timestamps(
    files: Sequence[Path], concatenate: bool
) -> npt.NDArray[np.int64] | list[npt.NDArray[np.int64]]:
    ts_buffer = []
    for time_file in files:
        with open(time_file, "rb") as f:
            ts_buffer.append(f.read())

    timestamps = [np.frombuffer(b, dtype="<u8").astype(np.int64) for b in ts_buffer]

    if concatenate:
        return np.concatenate(timestamps)
    else:
        return timestamps


class GrayFrame:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    @cached_property
    def bgr(self) -> npt.NDArray[np.uint8]:
        return (128 * np.ones([self.height, self.width, 3])).astype("uint8")

    @cached_property
    def gray(self) -> npt.NDArray[np.uint8]:
        return (128 * np.ones([self.height, self.width])).astype("unit8")
