from functools import cached_property
from pathlib import Path
from typing import Optional, Sequence, overload

import numpy as np
import numpy.typing as npt

import pupil_labs.video as plv
from pupil_labs.matching import MatchingMethod, SampledData, sample
from pupil_labs.neon_recording.frame import AudioFrame, VideoFrame
from pupil_labs.neon_recording.neon_timeseries import NeonTimeseries
from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_timestamps,
)
from pupil_labs.video import (
    ArrayLike,
    Indexer,
    MultiReader,
    Reader,
    ReaderFrameType,
    ReaderLike,
)
from pupil_labs.video.frame_slice import FrameSlice


class NeonVideoReader(MultiReader[ReaderFrameType], NeonTimeseries[ReaderFrameType]):
    def __init__(
        self,
        readers: Sequence[ReaderLike],
        timestamps: Sequence[npt.NDArray[np.int64]],
        rec_start: int,
    ) -> None:
        super().__init__(*readers)

        if len(self) != sum(len(t) for t in timestamps):
            raise ValueError("number of timestamps must match number of frames")

        self._timestamps = timestamps
        self._rec_start = rec_start

    @staticmethod
    def from_native_recording(
        rec_dir: Path, camera_name: str, rec_start: int
    ) -> "NeonVideoReader":
        files = find_sorted_multipart_files(rec_dir, camera_name, ".mp4")
        readers = [Reader(p[0]) for p in files]
        timestamps = load_multipart_timestamps([p[1] for p in files], concatenate=False)
        return NeonVideoReader(readers, timestamps, rec_start)

    @cached_property
    def audio(self) -> "NeonVideoReader[AudioFrame]":
        audio_readers = []
        abs_audio_timestamps = []
        for reader, ts in zip(self.readers, self._timestamps):
            if reader.audio is None:
                raise ValueError("not all readers have audio")
            audio_readers.append(reader.audio)

            audio_container_ts = (reader.audio.container_timestamps * 1e9).astype(
                np.int64
            )
            abs_duration = ts[-1] - ts[0]
            container_duration = audio_container_ts[-1] - audio_container_ts[0]
            sacling_factor = abs_duration / container_duration
            abs_ts = ts[0] + (audio_container_ts * sacling_factor).astype(np.int64)
            abs_audio_timestamps.append(abs_ts)

        return NeonVideoReader(audio_readers, abs_audio_timestamps, self._rec_start)

    @property
    def timestamps(self) -> npt.NDArray[np.int64]:
        """Absolute timestamps in nanoseconds."""
        return np.concatenate(self._timestamps)

    @cached_property
    def rel_timestamps(self) -> npt.NDArray[np.float64]:
        """Relative timestamps in seconds in relation to the recording  beginning."""
        return (self.timestamps - self._rec_start) / 1e9

    @property
    def by_abs_timestamp(self) -> Indexer[ReaderFrameType]:
        """Time-based access to video frames using absolute timestamps."""
        return Indexer(self.timestamps, self)

    @property
    def by_rel_timestamp(self) -> Indexer[ReaderFrameType]:
        """Time-based access to video frames using relative timestamps.

        Timestamps are relative to the beginning of the recording.
        """
        return Indexer(self.rel_timestamps, self)

    @overload
    def __getitem__(self, key: int) -> ReaderFrameType: ...
    @overload
    def __getitem__(
        self, key: slice | npt.NDArray[np.int64]
    ) -> FrameSlice[ReaderFrameType] | list[ReaderFrameType]: ...
    def __getitem__(
        self, key: int | slice | npt.NDArray[np.int64]
    ) -> ReaderFrameType | FrameSlice[ReaderFrameType] | list[ReaderFrameType]:
        if isinstance(key, int):
            frame = super().__getitem__(key)

            if isinstance(frame, plv.VideoFrame):
                frame = VideoFrame(
                    av_frame=frame.av_frame,
                    time=frame.time,
                    index=frame.index,
                    source=self,
                    timestamp=self.timestamps[key],
                )
                return frame
            elif isinstance(frame, plv.AudioFrame):
                frame = AudioFrame(
                    av_frame=frame.av_frame,
                    time=frame.time,
                    index=frame.index,
                    source=self,
                    timestamp=self.timestamps[key],
                )
                return frame
            else:
                raise TypeError(f"unsupported frame type: {type(frame)}")
        elif isinstance(key, slice):
            frameslice = FrameSlice[ReaderFrameType](
                self, key, lazy_frame_slice_limit=self.lazy_frame_slice_limit
            )
            if len(frameslice) < self.lazy_frame_slice_limit:
                return list(frameslice)
            return frameslice
        elif isinstance(key, np.ndarray):
            return [self[int(i)] for i in key]
        else:
            raise TypeError(f"unsupported key type: {type(key)}")

    def sample(
        self,
        timestamps: ArrayLike[int],
        method: MatchingMethod = MatchingMethod.NEAREST,
        tolerance: Optional[int] = None,
    ) -> SampledData[ReaderFrameType]:
        return sample(
            timestamps,
            self,
            method=method,
            tolerance=tolerance,
        )
