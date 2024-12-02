import json
import os
from functools import cached_property
from pathlib import Path
from typing import NamedTuple

import av
import av.video.frame
import av.audio.frame

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from pupil_labs.neon_recording import imu_pb2
from pupil_labs.video import MultiReader


def load_info(rec_dir: Path):
    with open(rec_dir / "info.json") as f:
        info = json.load(f)
    return info


def load_gaze_ground_truth(rec_dir: Path):
    info = load_info(rec_dir)
    gaze_200hz_file = rec_dir / "gaze_200hz.raw"
    if os.path.exists(gaze_200hz_file):
        time_200hz_file = rec_dir / "gaze_200hz.time"
        data = np.fromfile(gaze_200hz_file, "<f4").reshape([-1, 2])
        abs_ts = np.fromfile(time_200hz_file, dtype="<u8").astype(np.int64)
        rel_ts = (abs_ts - info["start_time"]) / 1e9
    else:
        raise NotImplementedError
    return (data, abs_ts, rel_ts)


class GazeGroundTruth(NamedTuple):
    abs_timestamp: npt.NDArray[np.int64]
    abs_ts: npt.NDArray[np.int64]
    rel_timestamp: npt.NDArray[np.float64]
    rel_ts: npt.NDArray[np.float64]
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    xy: npt.NDArray[np.float64]


class EyeStateGroundTruth(NamedTuple):
    abs_timestamp: npt.NDArray[np.int64]
    abs_ts: npt.NDArray[np.int64]
    rel_timestamp: npt.NDArray[np.float64]
    rel_ts: npt.NDArray[np.float64]
    pupil_diameter_left: npt.NDArray[np.float64]
    eyeball_center_left: npt.NDArray[np.float64]
    optical_axis_left: npt.NDArray[np.float64]
    pupil_diameter_right: npt.NDArray[np.float64]
    eyeball_center_right: npt.NDArray[np.float64]
    optical_axis_right: npt.NDArray[np.float64]


class IMUGroundTruth(NamedTuple):
    abs_timestamp: npt.NDArray[np.int64]
    abs_ts: npt.NDArray[np.int64]
    rel_timestamp: npt.NDArray[np.float64]
    rel_ts: npt.NDArray[np.float64]
    gyro: npt.NDArray[np.float64]
    accel: npt.NDArray[np.float64]
    euler: npt.NDArray[np.float64]
    quaternion: npt.NDArray[np.float64]


class EventGroundTruth(NamedTuple):
    abs_timestamp: npt.NDArray[np.int64]
    abs_ts: npt.NDArray[np.int64]
    rel_timestamp: npt.NDArray[np.float64]
    rel_ts: npt.NDArray[np.float64]
    event_name: npt.NDArray[np.str_]


class VideoGroundTruth(NamedTuple):
    abs_timestamp: npt.NDArray[np.int64]
    abs_ts: npt.NDArray[np.int64]
    rel_timestamp: npt.NDArray[np.float64]
    rel_ts: npt.NDArray[np.float64]


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
        for raw_file_path, time_file_path in zip(raw_file_paths, time_file_paths):
            d = np.fromfile(raw_file_path, "<f4").reshape([-1, 2])
            abs_ts = np.fromfile(time_file_path, dtype="<u8").astype(np.int64)
            data.append(d)
            abs_timestamp.append(abs_ts)
        data = np.concatenate(data)
        abs_timestamp = np.concatenate(abs_timestamp)
        rel_timestamp = (abs_timestamp - self.info["start_time"]) / 1e9

        return GazeGroundTruth(
            abs_timestamp=abs_timestamp,
            abs_ts=abs_timestamp,
            rel_timestamp=rel_timestamp,
            rel_ts=rel_timestamp,
            x=data[:, 0],
            y=data[:, 1],
            xy=data,
        )

    @cached_property
    def eye_state(self) -> EyeStateGroundTruth:
        raw_file_paths = sorted(self.rec_dir.glob("eye_state ps*.raw"))
        time_file_paths = sorted(self.rec_dir.glob("eye_state ps*.time"))
        assert len(raw_file_paths) == len(time_file_paths)

        data = []
        abs_timestamp = []
        for raw_file_path, time_file_path in zip(raw_file_paths, time_file_paths):
            d = np.fromfile(raw_file_path, "<f4").reshape([-1, 14])
            abs_ts = np.fromfile(time_file_path, dtype="<u8").astype(np.int64)
            data.append(d)
            abs_timestamp.append(abs_ts)
        data = np.concatenate(data)
        abs_timestamp = np.concatenate(abs_timestamp)
        rel_timestamp = (abs_timestamp - self.info["start_time"]) / 1e9

        return EyeStateGroundTruth(
            abs_timestamp=abs_timestamp,
            abs_ts=abs_timestamp,
            rel_timestamp=rel_timestamp,
            rel_ts=rel_timestamp,
            pupil_diameter_left=data[:, 0],
            eyeball_center_left=data[:, 1:4],
            optical_axis_left=data[:, 4:7],
            pupil_diameter_right=data[:, 7],
            eyeball_center_right=data[:, 8:11],
            optical_axis_right=data[:, 11:14],
        )

    @staticmethod
    def _parse_neon_imu_raw_packets(buffer):
        index = 0
        packet_sizes = []
        while True:
            nums = np.frombuffer(buffer[index : index + 2], np.uint16)

            if nums.size <= 0:
                break

            index += 2
            packet_size = int(nums[0])
            packet_sizes.append(packet_size)
            packet_bytes = buffer[index : index + packet_size]
            index += packet_size
            packet = imu_pb2.ImuPacket()  # type: ignore
            packet.ParseFromString(packet_bytes)

            yield packet

    @cached_property
    def imu(self) -> IMUGroundTruth:
        raw_file_paths = sorted(self.rec_dir.glob("extimu ps*.raw"))

        data = []
        abs_timestamp = []
        for raw_file_path in raw_file_paths:
            with raw_file_path.open("rb") as raw_file:
                raw_data = raw_file.read()
                imu_packets = self._parse_neon_imu_raw_packets(raw_data)

                time_data = []
                imu_data = []
                for packet in imu_packets:
                    time_data.append(packet.tsNs)

                    rotation = Rotation.from_quat(
                        [
                            packet.rotVecData.x,
                            packet.rotVecData.y,
                            packet.rotVecData.z,
                            packet.rotVecData.w,
                        ]
                    )

                    euler = rotation.as_euler(seq="XZY", degrees=True)

                    w = packet.rotVecData.w
                    x = packet.rotVecData.x
                    y = packet.rotVecData.y
                    z = packet.rotVecData.z
                    norms = np.sqrt(w**2 + x**2 + y**2 + z**2)

                    w /= norms
                    x /= norms
                    y /= norms
                    z /= norms

                    imu_data.append(
                        (
                            packet.gyroData.x,
                            packet.gyroData.y,
                            packet.gyroData.z,
                            packet.accelData.x,
                            packet.accelData.y,
                            packet.accelData.z,
                            *euler,
                            w,
                            x,
                            y,
                            z,
                        )
                    )

            d = np.array(imu_data)
            t = np.array(time_data)
            abs_timestamp.append(t)
            data.append(d)
        data = np.concatenate(data)
        abs_timestamp = np.concatenate(abs_timestamp)
        rel_timestamp = (abs_timestamp - self.info["start_time"]) / 1e9

        return IMUGroundTruth(
            abs_timestamp=abs_timestamp,
            abs_ts=abs_timestamp,
            rel_timestamp=rel_timestamp,
            rel_ts=rel_timestamp,
            gyro=data[:, 0:3],
            accel=data[:, 3:6],
            euler=data[:, 6:9],
            quaternion=data[:, 9:],
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
        rel_timestamp = (abs_timestamp - self.info["start_time"]) / 1e9

        return EventGroundTruth(
            abs_timestamp=abs_timestamp,
            abs_ts=abs_timestamp,
            rel_timestamp=rel_timestamp,
            rel_ts=rel_timestamp,
            event_name=data,
        )

    def _load_video(self, base_name: str):
        video_paths = sorted(self.rec_dir.glob(f"{base_name} ps*.mp4"))
        time_file_paths = sorted(self.rec_dir.glob(f"{base_name} ps*.time"))
        assert len(video_paths) == len(time_file_paths)

        reader = MultiReader(video_paths)

        abs_timestamp = []
        for time_file_path in time_file_paths:
            abs_ts = np.fromfile(time_file_path, dtype="<u8").astype(np.int64)
            abs_timestamp.append(abs_ts)
        abs_timestamp = np.concatenate(abs_timestamp)
        rel_timestamp = (abs_timestamp - self.info["start_time"]) / 1e9

        return VideoGroundTruth(
            abs_timestamp=abs_timestamp,
            abs_ts=abs_timestamp,
            rel_timestamp=rel_timestamp,
            rel_ts=rel_timestamp,
        )

    @property
    def eye(self) -> VideoGroundTruth:
        return self._load_video("Neon Sensor Module v1")

    @property
    def scene(self) -> VideoGroundTruth:
        return self._load_video("Neon Scene Camera v1")
