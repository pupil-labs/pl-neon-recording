import logging

import numpy as np

from pupil_labs.neon_recording.constants import (
    TIMESTAMP_DTYPE,
    TIMESTAMP_FIELD_NAME,
)
from pupil_labs.neon_recording.stream.array_record import Array, Record, fields

from ...utils import find_sorted_multipart_files, join_struct_arrays
from ..stream import Stream, StreamProps
from . import imu_pb2

log = logging.getLogger(__name__)


class ImuProps(StreamProps):
    gyro_xyz = fields[np.float64](["gyro_x", "gyro_y", "gyro_z"])
    "Gyroscope data"

    accel_xyz = fields[np.float64](["accel_x", "accel_y", "accel_z"])
    "Acceleration data"

    quaternion_wxyz = fields[np.float64]([
        "quaternion_w",
        "quaternion_x",
        "quaternion_y",
        "quaternion_z",
    ])
    "Orientation as a quaternion"


class ImuRecord(Record, ImuProps):
    def keys(self):
        keys = ImuProps.__dict__.keys()
        return [x for x in keys if not x.startswith("_")]


class ImuArray(Array[ImuRecord], ImuProps):
    record_class = ImuRecord


class IMUStream(Stream[ImuArray, ImuRecord], ImuProps):
    """Motion and orientation data"""

    FALLBACK_DTYPE = np.dtype([
        ("gyro_x", "float32"),
        ("gyro_y", "float32"),
        ("gyro_z", "float32"),
        ("accel_x", "float32"),
        ("accel_y", "float32"),
        ("accel_z", "float32"),
        ("quaternion_w", "float32"),
        ("quaternion_x", "float32"),
        ("quaternion_y", "float32"),
        ("quaternion_z", "float32"),
    ])

    def __init__(self, recording):
        log.debug("NeonRecording: Loading IMU data")

        imu_file_pairs = find_sorted_multipart_files(recording._rec_dir, "imu")

        if len(imu_file_pairs) > 0:
            imu_data = Array(  # type: ignore
                [file for file, _ in imu_file_pairs],
                fallback_dtype=np.dtype(IMUStream.FALLBACK_DTYPE),
            )
            imu_data.dtype.names = [  # type: ignore
                TIMESTAMP_FIELD_NAME if name == "timestamp_ns" else name
                for name in imu_data.dtype.names  # type: ignore
            ]

        else:
            imu_file_pairs = find_sorted_multipart_files(recording._rec_dir, "extimu")
            time_data = Array([file for _, file in imu_file_pairs], TIMESTAMP_DTYPE)  # type: ignore

            records = []
            for imu_file, _ in imu_file_pairs:
                with imu_file.open("rb") as raw_file:
                    raw_data = raw_file.read()
                    imu_packets = parse_neon_imu_raw_packets(raw_data)

                    records.extend([
                        (
                            packet.gyroData.x,
                            packet.gyroData.y,
                            packet.gyroData.z,
                            packet.accelData.x,
                            packet.accelData.y,
                            packet.accelData.z,
                            packet.rotVecData.w,
                            packet.rotVecData.x,
                            packet.rotVecData.y,
                            packet.rotVecData.z,
                        )
                        for packet in imu_packets
                    ])

            imu_data = np.array(records, dtype=IMUStream.FALLBACK_DTYPE)  # type: ignore
            imu_data = join_struct_arrays([time_data, imu_data])

        super().__init__("imu", recording, imu_data.view(ImuArray))


def parse_neon_imu_raw_packets(buffer):
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
