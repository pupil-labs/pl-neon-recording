from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    SupportsIndex,
    TypeVar,
    overload,
)

import numpy as np
import numpy.typing as npt

from pupil_labs.neon_recording.constants import TIMESTAMP_FIELD_NAME
from pupil_labs.neon_recording.sample import match_ts
from pupil_labs.neon_recording.timeseries.array_record import (
    Array,
    RecordType,
    fields,
)
from pupil_labs.video import ArrayLike

if TYPE_CHECKING:
    from ..neon_recording import NeonRecording

ArrayType = TypeVar("ArrayType", bound=Array)


class TimeseriesProps:
    ts: npt.NDArray[np.int64] = fields[np.int64](TIMESTAMP_FIELD_NAME)  # type:ignore
    "The moment these data were recorded"

    def keys(self):
        return dir(self)

    def __len__(self):
        return len(self.keys())


T = TypeVar("T", bound="Timeseries")


class Timeseries(TimeseriesProps, Generic[ArrayType, RecordType]):
    """Base class for all Neon timeseries data."""

    name: str
    recording: "NeonRecording"
    _data: ArrayType

    def __init__(self, data: ArrayType, name: str, recording: "NeonRecording"):
        self.name = name
        self.recording = recording
        self._data = data

    @property
    def dtype(self):
        return self._data.dtype

    def keys(self):
        if not self._data.dtype:
            return ["0"]
        assert self._data.dtype.names is not None
        return list(self._data.dtype.names)

    def __len__(self):
        return len(self._data)

    @overload
    def __getitem__(self, key: SupportsIndex) -> RecordType: ...
    @overload
    def __getitem__(self: T, key: slice | list[str]) -> T: ...
    def __getitem__(
        self, key: SupportsIndex | slice | str | list[str]
    ) -> "Timeseries | RecordType":
        if isinstance(key, slice):
            return self.__class__(
                self._data[key],  # type: ignore
                self.recording,
            )
        else:
            return self._data[key]  # type: ignore

    @property
    def data(self):
        """Data as a numpy array"""
        return self._data

    @property
    def pd(self):
        """Data as a pandas DataFrame"""
        return self._data.pd

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(name={self.name!r}, recording={self.recording!r}, data={self._data!r})"
        )

    def sample(
        self: T,
        target_ts: ArrayLike[int],
        method: Literal["nearest", "backward", "forward"] = "nearest",
        tolerance: int | None = None,
    ) -> T:
        indices = match_ts(target_ts, self.ts, method, tolerance)

        if True in np.isnan(indices):
            raise ValueError(
                "Failed to find matching samples for some samples.",
                [t for t, i in zip(target_ts, indices, strict=False) if np.isnan(i)],
            )

        return self.__class__(
            self._data[indices],  # type: ignore
            self.recording,
        )
