from functools import cached_property
from pathlib import Path
from typing import Iterator, NamedTuple, Optional, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial.transform import Rotation  # type: ignore

from pupil_labs.matching import MatchingMethod, SampledData
from pupil_labs.neon_recording.imu_pb2 import ImuPacket  # type: ignore
from pupil_labs.neon_recording.neon_timeseries import NeonTimeseries
from pupil_labs.neon_recording.utils import find_sorted_multipart_files
from pupil_labs.video import ArrayLike, Indexer


class IMURecord(NamedTuple):
    abs_timestamp: int
    rel_timestamp: float
    gyro: npt.NDArray[np.float64]
    accel: npt.NDArray[np.float64]
    euler: npt.NDArray[np.float64]
    quaternion: npt.NDArray[np.float64]
    "Quaternion in the order (w, x, y, z)"

    @property
    def abs_ts(self) -> int:
        return self.abs_timestamp

    @property
    def rel_ts(self) -> float:
        return self.rel_timestamp

    @property
    def data(self) -> npt.NDArray[np.float64]:
        return np.concatenate([
            [self.abs_ts],
            self.gyro,
            self.accel,
            self.euler,
            self.quaternion,
        ])


class IMU(NeonTimeseries[IMURecord]):
    def __init__(
        self, time_data: ArrayLike[int], data: ArrayLike[float], rec_start: int = 0
    ):
        self._time_data = np.array(time_data)
        self._data = np.array(data)
        self._rec_start = rec_start

    @staticmethod
    def from_native_recording(rec_dir: Path, rec_start: int) -> "IMU":
        imu_files = find_sorted_multipart_files(rec_dir, "extimu")
        imu_data = []
        ts_data = []

        for imu_file, _ in imu_files:
            with imu_file.open("rb") as raw_file:
                raw_data = raw_file.read()
                imu_packets = IMU._parse_neon_imu_raw_packets(raw_data)

                for packet in imu_packets:
                    rotation = Rotation.from_quat([
                        packet.rotVecData.x,
                        packet.rotVecData.y,
                        packet.rotVecData.z,
                        packet.rotVecData.w,
                    ])

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

                    ts_data.append(
                        packet.tsNs,
                    )
                    imu_data.append((
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
                    ))

        data = np.array(imu_data)
        ts = np.array(ts_data)
        return IMU(ts, data, rec_start)

    @property
    def abs_timestamp(self) -> npt.NDArray[np.int64]:
        return self._time_data

    abs_ts = abs_timestamp

    @cached_property
    def rel_timestamp(self) -> npt.NDArray[np.float64]:
        """Relative timestamps in seconds in relation to the recording beginning."""
        return (self.abs_timestamp - self._rec_start) / 1e9

    @property
    def rel_ts(self) -> npt.NDArray[np.float64]:
        return self.rel_timestamp

    @property
    def by_abs_timestamp(self) -> Indexer[IMURecord]:
        return Indexer(self.abs_timestamp, self)

    @property
    def by_rel_timestamp(self) -> Indexer[IMURecord]:
        return Indexer(self.rel_timestamp, self)

    @property
    def data(self) -> npt.NDArray[np.float64]:
        return self._data

    abs_ts = abs_timestamp

    @property
    def gyro(self) -> npt.NDArray[np.float64]:
        return self._data[:, 0:3]

    @property
    def accel(self) -> npt.NDArray[np.float64]:
        return self._data[:, 3:6]

    @property
    def euler(self) -> npt.NDArray[np.float64]:
        return self._data[:, 6:9]

    @property
    def quaternion(self) -> npt.NDArray[np.float64]:
        return self._data[:, 9:13]

    def __len__(self) -> int:
        return len(self._time_data)

    @overload
    def __getitem__(self, key: int, /) -> IMURecord: ...
    @overload
    def __getitem__(self, key: slice, /) -> "IMU": ...
    def __getitem__(self, key: int | slice) -> "IMURecord | IMU":
        if isinstance(key, int):
            record = IMURecord(
                self.abs_timestamp[key],
                self.rel_timestamp[key],
                self._data[key, 0:3],
                self._data[key, 3:6],
                self._data[key, 6:9],
                self._data[key, 9:13],
            )
            return record
        elif isinstance(key, slice):
            return IMU(self._time_data[key], self._data[key], self._rec_start)
        else:
            raise TypeError(f"Invalid argument type {type(key)}")

    def __iter__(self) -> Iterator[IMURecord]:
        for i in range(len(self)):
            yield self[i]

    def sample(
        self,
        timestamps: ArrayLike[int],
        method: MatchingMethod = MatchingMethod.NEAREST,
        tolerance: Optional[int] = None,
    ) -> SampledData[IMURecord]:
        return SampledData.sample(
            timestamps,
            self,
            method=method,
            tolerance=tolerance,
        )

    def interpolate(self, timestamps: ArrayLike[int]) -> "IMU":
        timestamps = np.array(timestamps)
        interp_data = []

        for key in [
            "gyro",
            "accel",
            "euler",
            "quaternion",
        ]:
            data_source = getattr(self, key)

            for dim in range(data_source.shape[1]):
                interp_dim = np.interp(
                    timestamps, self.abs_timestamp, data_source[:, dim]
                )
                interp_data.append(interp_dim)
        interp_arr = np.column_stack(interp_data)
        return IMU(timestamps, interp_arr, self._rec_start)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._data,
            columns=[
                "gyro_x",
                "gyro_y",
                "gyro_z",
                "accel_x",
                "accel_y",
                "accel_z",
                "euler_x",
                "euler_y",
                "euler_z",
                "quaternion_w",
                "quaternion_x",
                "quaternion_y",
                "quaternion_z",
            ],
            index=self._time_data,
        )

    @staticmethod
    def _parse_neon_imu_raw_packets(buffer: bytes) -> Iterator[ImuPacket]:  # type: ignore
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
            packet = ImuPacket()
            packet.ParseFromString(packet_bytes)

            yield packet
