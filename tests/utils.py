import json
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pupil_labs.neon_recording.timeseries.imu.imu_timeseries import (
    parse_neon_imu_raw_packets,
)


def load_info(rec_dir: Path):
    with open(rec_dir / "info.json") as f:
        info = json.load(f)
    return info


@dataclass
class SensorGroundTruth:
    time: npt.NDArray[np.int64]


@dataclass
class GazeGroundTruth(SensorGroundTruth):
    point: npt.NDArray[np.float32]


@dataclass
class EyeballGroundTruth(SensorGroundTruth):
    center_left: npt.NDArray[np.float32]
    optical_axis_left: npt.NDArray[np.float32]
    center_right: npt.NDArray[np.float32]
    optical_axis_right: npt.NDArray[np.float32]


@dataclass
class PupilGroundTruth(SensorGroundTruth):
    diameter_left: npt.NDArray[np.float32]
    diameter_right: npt.NDArray[np.float32]


@dataclass
class EyelidGroundTruth(SensorGroundTruth):
    angle_left: npt.NDArray[np.float32]
    aperture_left: npt.NDArray[np.float32]
    angle_right: npt.NDArray[np.float32]
    aperture_right: npt.NDArray[np.float32]


@dataclass
class IMUGroundTruth(SensorGroundTruth):
    time: npt.NDArray[np.int64]
    angular_velocity: npt.NDArray[np.float64]
    acceleration: npt.NDArray[np.float64]
    rotation: npt.NDArray[np.float64]


@dataclass
class EventGroundTruth(SensorGroundTruth):
    event: npt.NDArray[np.str_]


@dataclass
class AVGroundTruth(SensorGroundTruth): ...


class GroundTruth:
    def __init__(self, rec_dir: Path):
        self.rec_dir = rec_dir

    @property
    def info(self):
        with open(self.rec_dir / "info.json") as f:
            return json.load(f)

    @cached_property
    def gaze(self) -> GazeGroundTruth:
        gaze_200hz_file = self.rec_dir / "gaze_200hz.raw"
        if os.path.exists(gaze_200hz_file):
            time_200hz_file = self.rec_dir / "gaze_200hz.time"
            raw_file_paths = [gaze_200hz_file]
            time_file_paths = [time_200hz_file]
        else:
            raw_file_paths = sorted(self.rec_dir.glob("gaze ps*.raw"))
            time_file_paths = sorted(self.rec_dir.glob("gaze ps*.time"))
            assert len(raw_file_paths) == len(time_file_paths)

        data = []
        abs_timestamp = []
        for raw_file_path, time_file_path in zip(
            raw_file_paths, time_file_paths, strict=False
        ):
            d = np.fromfile(raw_file_path, "<f4").reshape([-1, 2])
            abs_ts = np.fromfile(time_file_path, dtype="<u8").astype(np.int64)
            data.append(d)
            abs_timestamp.append(abs_ts)
        data = np.concatenate(data)
        abs_timestamp = np.concatenate(abs_timestamp)

        return GazeGroundTruth(
            time=abs_timestamp,
            point=data,
        )

    @cached_property
    def eyeball(self) -> EyeballGroundTruth:
        raw_file_paths = sorted(self.rec_dir.glob("eye_state ps*.raw"))
        time_file_paths = sorted(self.rec_dir.glob("eye_state ps*.time"))
        assert len(raw_file_paths) == len(time_file_paths)

        data = []
        abs_timestamp = []
        for raw_file_path, time_file_path in zip(
            raw_file_paths, time_file_paths, strict=False
        ):
            d = np.fromfile(raw_file_path, "<f4").reshape([-1, 20])
            abs_ts = np.fromfile(time_file_path, dtype="<i8").astype(np.int64)
            data.append(d)
            abs_timestamp.append(abs_ts)
        data = np.concatenate(data)
        abs_timestamp = np.concatenate(abs_timestamp)

        return EyeballGroundTruth(
            time=abs_timestamp,
            center_left=data[:, 1:4],
            optical_axis_left=data[:, 4:7],
            center_right=data[:, 8:11],
            optical_axis_right=data[:, 11:14],
        )

    @cached_property
    def pupil(self) -> PupilGroundTruth:
        raw_file_paths = sorted(self.rec_dir.glob("eye_state ps*.raw"))
        time_file_paths = sorted(self.rec_dir.glob("eye_state ps*.time"))
        assert len(raw_file_paths) == len(time_file_paths)

        data = []
        abs_timestamp = []
        for raw_file_path, time_file_path in zip(
            raw_file_paths, time_file_paths, strict=False
        ):
            d = np.fromfile(raw_file_path, "<f4").reshape([-1, 20])
            abs_ts = np.fromfile(time_file_path, dtype="<i8").astype(np.int64)
            data.append(d)
            abs_timestamp.append(abs_ts)
        data = np.concatenate(data)
        abs_timestamp = np.concatenate(abs_timestamp)

        return PupilGroundTruth(
            time=abs_timestamp,
            diameter_left=data[:, 0],
            diameter_right=data[:, 7],
        )

    @cached_property
    def eyelid(self) -> EyelidGroundTruth:
        raw_file_paths = sorted(self.rec_dir.glob("eye_state ps*.raw"))
        time_file_paths = sorted(self.rec_dir.glob("eye_state ps*.time"))
        assert len(raw_file_paths) == len(time_file_paths)

        data = []
        abs_timestamp = []
        for raw_file_path, time_file_path in zip(
            raw_file_paths, time_file_paths, strict=False
        ):
            d = np.fromfile(raw_file_path, "<f4").reshape([-1, 20])
            abs_ts = np.fromfile(time_file_path, dtype="<i8").astype(np.int64)
            data.append(d)
            abs_timestamp.append(abs_ts)
        data = np.concatenate(data)
        abs_timestamp = np.concatenate(abs_timestamp)

        return EyelidGroundTruth(
            time=abs_timestamp,
            angle_left=data[:, 14:16],
            aperture_left=data[:, 16],
            angle_right=data[:, 17:19],
            aperture_right=data[:, 19],
        )

    @cached_property
    def imu(self) -> IMUGroundTruth:
        raw_file_paths = sorted(self.rec_dir.glob("extimu ps*.raw"))

        data = []
        abs_timestamp = []
        for raw_file_path in raw_file_paths:
            with raw_file_path.open("rb") as raw_file:
                raw_data = raw_file.read()
                imu_packets = parse_neon_imu_raw_packets(raw_data)

                time_data = []
                imu_data = []
                for packet in imu_packets:
                    time_data.append(packet.tsNs)

                    w = packet.rotVecData.w
                    x = packet.rotVecData.x
                    y = packet.rotVecData.y
                    z = packet.rotVecData.z

                    imu_data.append((
                        packet.gyroData.x,
                        packet.gyroData.y,
                        packet.gyroData.z,
                        packet.accelData.x,
                        packet.accelData.y,
                        packet.accelData.z,
                        w,
                        x,
                        y,
                        z,
                    ))

            d = np.array(imu_data, dtype=np.float32)
            t = np.array(time_data)
            abs_timestamp.append(t)
            data.append(d)
        data = np.concatenate(data)
        abs_timestamp = np.concatenate(abs_timestamp)

        rotation_w = data[:, 6]
        rotation_xyz = data[:, 7:]
        rotation_xyzw = np.column_stack((rotation_xyz, rotation_w))
        return IMUGroundTruth(
            time=abs_timestamp,
            angular_velocity=data[:, 0:3],
            acceleration=data[:, 3:6],
            rotation=rotation_xyzw,
        )

    @cached_property
    def events(self) -> EventGroundTruth:
        events_file = self.rec_dir / "event.txt"
        time_file = self.rec_dir / "event.time"

        with open(events_file) as f:
            events = f.readlines()
            events = [event.strip() for event in events]

        abs_timestamp = np.fromfile(time_file, dtype="<u8").astype(np.int64)
        data = np.array(events)

        return EventGroundTruth(
            time=abs_timestamp,
            event=data,
        )

    def _load_video(self, base_name: str):
        video_paths = sorted(self.rec_dir.glob(f"{base_name} ps*.mp4"))
        time_file_paths = sorted(self.rec_dir.glob(f"{base_name} ps*.time"))
        assert len(video_paths) == len(time_file_paths)

        abs_timestamp = []
        for time_file_path in time_file_paths:
            abs_ts = np.fromfile(time_file_path, dtype="<u8").astype(np.int64)
            abs_timestamp.append(abs_ts)
        abs_timestamp = np.concatenate(abs_timestamp)

        return AVGroundTruth(time=abs_timestamp)

    def _load_audio(self, base_name: str):
        video_paths = sorted(self.rec_dir.glob(f"{base_name} ps*.mp4"))
        time_file_paths = sorted(self.rec_dir.glob(f"{base_name} ps*.time"))
        assert len(video_paths) == len(time_file_paths)

        abs_timestamp = []
        for time_file_path in time_file_paths:
            abs_ts = np.fromfile(time_file_path, dtype="<u8").astype(np.int64)
            abs_timestamp.append(abs_ts)
        abs_timestamp = np.concatenate(abs_timestamp)

        return AVGroundTruth(time=abs_timestamp)

    @property
    def eye(self) -> AVGroundTruth:
        return self._load_video("Neon Sensor Module v1")

    @property
    def scene(self) -> AVGroundTruth:
        return self._load_video("Neon Scene Camera v1")

    @property
    def audio(self) -> AVGroundTruth:
        return self._load_audio("Neon Scene Camera v1")
