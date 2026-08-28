import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pupil_labs.neon_recording.constants import TIMESTAMP_DTYPE
from pupil_labs.neon_recording.timeseries.array_record import Array

log = logging.getLogger(__name__)


# More than 1% of out-of-order timestamps indicates an issue that sorting
# alone might not solve
OUT_OF_ORDER_TS_THRESHOLD = 0.01

# Events might have custom timestamps that we have no control over
OUT_OF_ORDER_KINDS = ["event"]


def find_sorted_multipart_files(
    recording_path: Path, basename: str, extension: str = ".raw"
):
    basenames_without_times = ["worn", "gaze_left", "gaze_right"]
    file_pairs = []
    for raw_file in recording_path.glob(f"{basename} ps*{extension}"):
        if basename in basenames_without_times:
            # Some sensors don't have their own .time file
            gaze_time_stem = raw_file.stem.replace(f"{basename} ", "gaze ")
            time_file = raw_file.parent / f"{gaze_time_stem}.time"

        else:
            time_file = raw_file.with_suffix(".time")

        if time_file.exists():
            file_pairs.append((raw_file, time_file))

    return sorted(file_pairs, key=lambda pair: int(pair[0].stem[len(basename) + 3 :]))


def load_multipart_data_time_pairs(file_pairs, dtype):
    ts_files = [time_file for _, time_file in file_pairs]
    data_files = [data_file for data_file, _ in file_pairs]

    time_data = Array(ts_files, TIMESTAMP_DTYPE)  # type: ignore
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

    shortest = min(len(time_data), len(item_data))
    merged = join_struct_arrays([time_data[:shortest], item_data[:shortest]])
    return merged


def load_and_convert_tstamps(path: Path):
    return np.frombuffer(path.open("rb").read(), dtype="<i8")


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


def fix_timestamps(array: npt.NDArray, kind: str, column: str = "time") -> npt.NDArray:
    array = drop_zero_timestamps(array, kind, column)
    sort_timestamps(array, kind, column)
    return array


def drop_zero_timestamps(array: npt.NDArray, kind: str, column: str = "time") -> None:
    zero_mask = array[column] == 0
    if not np.any(zero_mask):
        return array

    num_zero = zero_mask.sum()
    num_total = len(array)
    percentage = num_zero / num_total
    log.warning(
        f"{num_zero} out of {num_total} timestamps ({percentage:.2%}) are set to 0 "
        f"in the `{kind}` timeseries, dropping them along with the corresponding data."
    )

    return array[~zero_mask]


def sort_timestamps(array: npt.NDArray, kind: str, column: str = "time") -> None:
    values = array[column]
    if values.size < 2:
        return

    # NOTE: checking for strictly increasing timestamps below
    previous_max = np.maximum.accumulate(values[:-1])
    out_of_order = np.flatnonzero(np.r_[False, values[1:] <= previous_max])
    if not out_of_order.size:
        return

    num_out_of_order = len(out_of_order)
    num_total = len(array)
    percentage = num_out_of_order / num_total

    verdict = ""
    if percentage > OUT_OF_ORDER_TS_THRESHOLD and kind not in OUT_OF_ORDER_KINDS:
        verdict = (
            " With such a high fraction of out-of-order timestamps, sorting might "
            f"not be sufficient to fix this issue, the `{kind}` data appear corrupted."
        )
    percentage_desc = f"{percentage:.2%}" if percentage >= 0.0001 else "<0.01%"
    log.warning(
        f"{num_out_of_order} out of {num_total} timestamps ({percentage_desc}) "
        f"appear in a non-increasing order in the `{kind}` timeseries. "
        f"The data were sorted to make timestamps monotonic.{verdict}"
    )

    array.sort(order=column)
