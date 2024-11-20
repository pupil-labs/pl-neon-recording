from pathlib import Path
from typing import Iterator, NamedTuple, Optional, overload

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from pupil_labs.matching import MatchingMethod, SampledData, sample
from pupil_labs.neon_recording.imu_pb2 import ImuPacket  # type: ignore
from pupil_labs.neon_recording.neon_timeseries import NeonTimeseries
from pupil_labs.neon_recording.utils import find_sorted_multipart_files
from pupil_labs.video.array_like import ArrayLike


class IMURecord(NamedTuple):
    ts: int
    gyro: npt.NDArray[np.float64]
    accel: npt.NDArray[np.float64]
    euler: npt.NDArray[np.float64]
    quaternion: npt.NDArray[np.float64]

    @property
    def data(self) -> npt.NDArray[np.float64]:
        return np.concatenate([
            [self.ts],
            self.gyro,
            self.accel,
            self.euler,
            self.quaternion,
        ])


class IMU(NeonTimeseries[IMURecord]):
    def __init__(self, time_data: ArrayLike[int], data: ArrayLike[float]):
        self._time_data = np.array(time_data)
        self._data = np.array(data)

    @staticmethod
    def from_native_recording(rec_dir: Path) -> "IMU":
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
        return IMU(ts, data)

    @property
    def timestamps(self) -> npt.NDArray[np.int64]:
        return self._time_data

    @property
    def data(self) -> npt.NDArray[np.float64]:
        return self._data

    ts = timestamps

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
                self._time_data[key],
                self._data[key, 0:3],
                self._data[key, 3:6],
                self._data[key, 6:9],
                self._data[key, 9:13],
            )
            return record
        elif isinstance(key, slice):
            return IMU(
                self._time_data[key],
                self._data[key],
            )
        else:
            raise TypeError(f"Invalid argument type {type(key)}")

    def __iter__(self) -> Iterator[IMURecord]:
        for i in range(len(self)):
            yield self[i]

    def sample(
        self,
        timestamps: ArrayLike[int],
        method: MatchingMethod = MatchingMethod.NEAREST,
        tolerance: Optional[float] = None,
    ) -> SampledData:
        return sample(
            timestamps,
            self,
            method=method,
            tolerance=tolerance,
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
            packet = ImuPacket()
            packet.ParseFromString(packet_bytes)

            yield packet
