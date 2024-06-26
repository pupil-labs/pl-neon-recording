import numpy as np

from ... import structlog
from .av_load import _load_av_container
from ..stream import Stream

log = structlog.get_logger(__name__)


def _convert_audio_data_to_recarray(audio_data, ts):
    log.debug("NeonRecording: Converting audio data to recarray format.")

    if audio_data.shape[0] != len(ts):
        log.error("NeonRecording: Length mismatch - audio_data and ts.")
        raise ValueError("audio_data and ts must have the same length")

    out = np.recarray(
        audio_data.shape[0],
        dtype=[("sample", "<f8"), ("ts", "<f8")],
    )
    out.sample = audio_data[:]
    out.ts = ts.astype(np.float64)

    return out


class AudioStream(Stream):
    def __init__(self, name, file_name, recording, container=None, video_ts=None):
        super().__init__(name, recording)
        self._file_name = file_name
        self._backing_container = container
        self._video_ts = video_ts
        self._sample_rate = None
        self._n_samples = None

        self._load()

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def n_samples(self):
        return self._n_samples

    def _load(self):
        # if a backing_container is supplied, then a ts array is usually also supplied
        if self._backing_container is None:
            log.info(f"NeonRecording: Loading audio from: {self._file_name}.")
            self._backing_container, self._video_ts = _load_av_container(
                self._recording._rec_dir, self._file_name
            )

        self._sample_rate = self._backing_container.streams.audio[0].sample_rate
        self._n_frames = self._backing_container.streams.audio[0].frames
        self._samples_per_frame = self._backing_container.streams.audio[0].frames[0].samples

        ac = 0
        audio_data = np.zeros(
            shape=self._samples_per_frame * (self._n_frames - 1), dtype=np.float64
        )
        sample_start_times = np.zeros(shape=(self._n_frames - 1), dtype=np.float64)
        for sc, sample in enumerate(self._backing_container.streams.audio[0].frames):
            sample_start_times[sc] = sample.time

            for val in sample.to_ndarray()[0]:
                audio_data[ac] = val
                ac += 1

        # rewind audio back to start
        self._backing_container.streams.audio[0].seek(0)

        self._ts = np.arange(self._video_ts[0], self._video_ts[-1], self._n_frames)
        audio_data = _convert_audio_data_to_recarray(audio_data, self._ts)

        self._backing_data = audio_data
        self._data = audio_data[:]

    def _sample_linear_interp(self, sorted_ts):
        pass

        samples = self._data.sample

        interp_data = np.zeros(
            len(sorted_ts),
            dtype=[("sample", "<f8"), ("ts", "<f8")],
        ).view(np.recarray)
        interp_data.sample = np.interp(sorted_ts, self._ts, samples, left=np.nan, right=np.nan)
        interp_data.ts = sorted_ts

        for d in interp_data:
            if not np.isnan(d.x):
                yield d
            else:
                yield None

