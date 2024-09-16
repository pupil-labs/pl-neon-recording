import numpy as np

from ... import structlog
from ..stream import Stream, SimpleDataSampler
from ...utils import find_sorted_multipart_files
from scipy.spatial.transform import Rotation
from . import imu_pb2

log = structlog.get_logger(__name__)


class IMUStreamSampler(SimpleDataSampler):
    def __init__(self, data):
        # Ensure that quaternions are normalized
        w = data['quaternion_w']
        x = data['quaternion_x']
        y = data['quaternion_y']
        z = data['quaternion_z']
        norms = np.sqrt(w**2 + x**2 + y**2 + z**2)

        data['quaternion_w'] /= norms
        data['quaternion_x'] /= norms
        data['quaternion_y'] /= norms
        data['quaternion_z'] /= norms

        super().__init__(data)


IMUStreamSampler.sampler_class = IMUStreamSampler


class IMUStream(Stream):
    """
        Motion and orientation data

        Each record contains:
            * `ts`: The moment these data were recorded
            * Gyroscope data
                * `gyro_x`
                * `gyro_y`
                * `gyro_z`
            * Acceleration data
                * `accel_x`
                * `accel_y`
                * `accel_z`
            * Orientation in Euler angles (degrees)
                * `pitch`
                * `yaw`
                * `roll`
            * Orientation as a quaternion
                * `quaternion_w`
                * `quaternion_x`
                * `quaternion_y`
                * `quaternion_z`
    """
    _DTYPE_RAW = np.dtype([
        ("ts", "<f8"),
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
    ])

    sampler_class = IMUStreamSampler

    def __init__(self, recording):
        log.debug("NeonRecording: Loading IMU data")

        imu_files = find_sorted_multipart_files(recording._rec_dir, "extimu")
        imu_data = []

        for imu_file, _ in imu_files:
            with imu_file.open("rb") as raw_file:
                raw_data = raw_file.read()
                imu_packets = _parse_neon_imu_raw_packets(raw_data)

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

                    imu_data.append((
                        packet.tsNs * 1e-9,
                        packet.gyroData.x, packet.gyroData.y, packet.gyroData.z,
                        packet.accelData.x, packet.accelData.y, packet.accelData.z,
                        *euler,
                        packet.rotVecData.w, packet.rotVecData.x, packet.rotVecData.y, packet.rotVecData.z,
                    ))

        data = np.array(imu_data, dtype=IMUStream._DTYPE_RAW).view(np.recarray)
        super().__init__("imu", recording, data)


def _parse_neon_imu_raw_packets(buffer):
    index = 0
    packet_sizes = []
    while True:
        nums = np.frombuffer(buffer[index: index + 2], np.uint16)

        if nums.size <= 0:
            break

        index += 2
        packet_size = int(nums[0])
        packet_sizes.append(packet_size)
        packet_bytes = buffer[index: index + packet_size]
        index += packet_size
        packet = imu_pb2.ImuPacket()
        packet.ParseFromString(packet_bytes)

        yield packet
