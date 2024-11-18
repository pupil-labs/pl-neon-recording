from functools import cached_property
from pathlib import Path
from typing import Sequence, overload

import numpy as np
import numpy.typing as npt

import pupil_labs.video as plv
from pupil_labs.video import Indexer, MultiReader, Reader, ReaderFrameType, ReaderLike
from pupil_labs.video.frame_slice import FrameSlice

from .frame import AudioFrame, VideoFrame
from .utils import find_sorted_multipart_files, load_multipart_timestamps


class VideoReader(MultiReader[ReaderFrameType]):
    def __init__(
        self,
        readers: Sequence[ReaderLike],
        timestamps: Sequence[npt.NDArray[np.int64]],
    ) -> None:
        super().__init__(*readers)

        if len(self) != sum(len(t) for t in timestamps):
            raise ValueError("number of timestamps must match number of frames")

        self._timestamps = timestamps

    @staticmethod
    def from_native_recording(rec_dir: Path, camera_name: str) -> "VideoReader":
        files = find_sorted_multipart_files(rec_dir, camera_name, ".mp4")
        readers = [Reader(p[0]) for p in files]
        timestamps = load_multipart_timestamps([p[1] for p in files], concatenate=False)
        return VideoReader(readers, timestamps)

    @cached_property
    def audio(self) -> "VideoReader[AudioFrame]":
        audio_readers = []
        audio_timestamps = []
        for reader, ts in zip(self.readers, self._timestamps):
            if reader.audio is None:
                raise ValueError("not all readers have audio")
            audio_readers.append(reader.audio)

            audio_pts = (reader.audio.video_time * 1e9).astype(np.int64)
            audio_ts = ts[0] + audio_pts
            audio_timestamps.append(audio_ts)

        return VideoReader(audio_readers, audio_timestamps)

    @cached_property
    def timestamps(self) -> npt.NDArray[np.int64]:
        return np.concatenate(self._timestamps)

    @property
    def by_timestamp(self) -> Indexer[ReaderFrameType]:
        return Indexer(self.timestamps, self)

    @overload
    def __getitem__(self, key: int) -> ReaderFrameType: ...
    @overload
    def __getitem__(
        self, key: slice
    ) -> FrameSlice[ReaderFrameType] | list[ReaderFrameType]: ...

    def __getitem__(
        self, key: int | slice
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
        else:
            frameslice = FrameSlice[ReaderFrameType](
                self, key, lazy_frame_slice_limit=self.lazy_frame_slice_limit
            )
            if len(frameslice) < self.lazy_frame_slice_limit:
                return list(frameslice)
            return frameslice
