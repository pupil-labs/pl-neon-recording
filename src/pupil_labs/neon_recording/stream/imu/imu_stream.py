import numpy as np

from ... import structlog
from ..stream import Stream
from ...utils import find_sorted_multipart_files, load_multipart_timestamps
from scipy.spatial.transform import Rotation
from . import imu_pb2

log = structlog.get_logger(__name__)


class IMUStream(Stream):
    DTYPE_RAW = np.dtype([
        ("gyro_x", "<f4"),
        ("gyro_y", "<f4"),
        ("gyro_z", "<f4"),
        ("accel_x", "<f4"),
        ("accel_y", "<f4"),
        ("accel_z", "<f4"),
        ("pitch", "<f4"),
        ("yaw", "<f4"),
        ("roll", "<f4"),
        ("quaternion_w", "<f4"),
        ("quaternion_x", "<f4"),
        ("quaternion_y", "<f4"),
        ("quaternion_z", "<f4"),
        ("tsNs", "uint64"),
        ("ts", "<f8"),
    ])

    def __init__(self, name, recording):
        log.info("NeonRecording: Loading IMU data")

        imu_files = find_sorted_multipart_files(recording._rec_dir, "extimu")
        imu_data = []

        for imu_file, _ in imu_files:
            with imu_file.open("rb") as raw_file:
                raw_data = raw_file.read()
                imu_packets = parse_neon_imu_raw_packets(raw_data)

                for packet in imu_packets:
                    rotation = Rotation.from_quat(
                        [
                            packet.rotVecData.x,
                            packet.rotVecData.y,
                            packet.rotVecData.z,
                            packet.rotVecData.w,
                        ]
                    )
                    euler = rotation.as_euler(seq="XZY", degrees=True)

                    ts = packet.tsNs

                    imu_data.append((
                        packet.gyroData.x, packet.gyroData.y, packet.gyroData.z,
                        packet.accelData.x, packet.accelData.y, packet.accelData.z,
                        *euler,
                        packet.rotVecData.w, packet.rotVecData.x, packet.rotVecData.y, packet.rotVecData.z,
                        packet.tsNs,
                        ts,
                    ))

        data = np.array(imu_data, dtype=IMUStream.DTYPE_RAW).view(np.recarray)
        super().__init__(name, recording, data)


def parse_neon_imu_raw_packets(buffer):
    index = 0
    packet_sizes = []
    while True:
        nums = np.frombuffer(buffer[index : index + 2], np.uint16)

        if nums.size <= 0:
            break

        index += 2
        packet_size = nums[0]
        packet_sizes.append(packet_size)
        packet_bytes = buffer[index : index + packet_size]
        index += packet_size
        packet = imu_pb2.ImuPacket()
        packet.ParseFromString(packet_bytes)

        yield packet
