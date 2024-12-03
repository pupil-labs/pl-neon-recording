from functools import cached_property
from pathlib import Path
from typing import Sequence, overload

import numpy as np
import numpy.typing as npt

import pupil_labs.video as plv
from pupil_labs.neon_recording.frame import AudioFrame, VideoFrame
from pupil_labs.neon_recording.frame_slice import FrameSlice
from pupil_labs.neon_recording.neon_timeseries import NeonTimeseries
from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_timestamps,
)
from pupil_labs.video import (
    MultiReader,
    Reader,
    ReaderFrameType,
    ReaderLike,
)


# TODO: Make this a NeonTimeseries
class NeonVideoReader(MultiReader[ReaderFrameType], NeonTimeseries):
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
        audio_readers: list[Reader[plv.AudioFrame]] = []
        abs_audio_timestamps: list[npt.NDArray[np.int64]] = []
        for reader, ts in zip(self.readers, self._timestamps):
            if reader.audio is None:
                raise ValueError("not all readers have audio")
            audio_readers.append(reader.audio)

            audio_container_ts = (reader.audio.container_timestamps * 1e9).astype(
                np.int64
            )
            abs_duration: int = ts[-1] - ts[0]
            container_duration: int = audio_container_ts[-1] - audio_container_ts[0]
            sacling_factor = abs_duration / container_duration
            abs_ts: npt.NDArray[np.int64] = ts[0] + (
                audio_container_ts * sacling_factor
            ).astype(np.int64)
            abs_audio_timestamps.append(abs_ts)
        audio_reader: NeonVideoReader[AudioFrame] = NeonVideoReader(
            audio_readers, abs_audio_timestamps, self._rec_start
        )
        return audio_reader

    @property
    def abs_timestamp(self) -> npt.NDArray[np.int64]:
        """Absolute timestamps in nanoseconds."""
        return np.concatenate(self._timestamps)

    @cached_property
    def rel_timestamp(self) -> npt.NDArray[np.float64]:
        """Relative timestamps in seconds in relation to the recording beginning."""
        return (self.abs_timestamp - self._rec_start) / 1e9

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
            plv_frame = super().__getitem__(key)
            if isinstance(plv_frame, plv.VideoFrame):
                video_frame = VideoFrame(
                    av_frame=plv_frame.av_frame,
                    time=plv_frame.time,
                    index=plv_frame.index,
                    source=self,
                    abs_timestamp=self.abs_timestamp[key],
                    rel_timestamp=self.rel_timestamp[key],
                )
                return video_frame
            elif isinstance(plv_frame, plv.AudioFrame):
                audio_frame = AudioFrame(
                    av_frame=plv_frame.av_frame,
                    time=plv_frame.time,
                    index=plv_frame.index,
                    source=self,
                    abs_timestamp=self.abs_timestamp[key],
                    rel_timestamp=self.rel_timestamp[key],
                )
                return audio_frame
            else:
                raise TypeError(f"unsupported frame type: {type(plv_frame)}")

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
