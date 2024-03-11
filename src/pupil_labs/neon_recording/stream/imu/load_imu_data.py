# adapted from @dom:
# https://github.com/pupil-labs/neon-player/blob/master/pupil_src/shared_modules/imu_timeline/imu_timeline.py#L72

import pathlib

from progress.spinner import LineSpinner
import numpy as np
from scipy.spatial.transform import Rotation

from . import imu_pb2
from ...time_utils import ns_to_s, load_and_convert_tstamps

from ... import structlog
log = structlog.get_logger(__name__)

def parse_neon_imu_raw_packets(buffer):
    index = 0
    packet_sizes = []
    while True:
        nums = np.frombuffer(buffer[index : index + 2], np.uint16)

        # TODO
        # DeprecationWarning: The truth value of an empty array is ambiguous.
        # Returning False, but in future this will result in an error.
        # Use `array.size > 0` to check that an array is not empty.
        if not nums:
            break

        index += 2
        packet_size = nums[0]
        packet_sizes.append(packet_size)
        packet_bytes = buffer[index : index + packet_size]
        index += packet_size
        packet = imu_pb2.ImuPacket()
        packet.ParseFromString(packet_bytes)
        yield packet


class IMURecording:
    DTYPE_RAW = np.dtype(
        [
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
            ("ts_rel", "<f8")
        ]
    )

    def __init__(self, path_to_imu_raw: pathlib.Path, start_ts: float) -> None:
        stem = path_to_imu_raw.stem
        self.path_raw = path_to_imu_raw
        # self.path_ts = path_to_imu_raw.with_name(stem + "_timestamps.npy")
        self.path_ts = path_to_imu_raw.with_name(stem + ".time")
        self.load(start_ts)


    def load(self, start_ts):
        log.debug("NeonRecording: Loading IMU data.")

        if not self.path_raw.exists() and self.path_ts.exists():
            # warnings.warn(f"IMU data not found at {self.path_raw.name} or error occurred when converting IMU timestamps.")
            log.warn(f"IMU data not found at {self.path_raw.name} or error occurred when converting IMU timestamps.")

            self.ts = np.empty(0, dtype=np.float64)
            self.raw = []
            return

        # self.ts = np.load(str(self.path_ts))
        self.ts = load_and_convert_tstamps(self.path_ts)
        with self.path_raw.open('rb') as raw_file:
            raw_data = raw_file.read()
            imu_packets = parse_neon_imu_raw_packets(raw_data)
            imu_data = []

            log.debug("NeonRecording: Iterating and converting IMU packets.")

            spinner = LineSpinner('Loading IMU data... ')

            # for packet in tqdm.tqdm(imu_packets):
            for packet in imu_packets:
                rotation = Rotation.from_quat([packet.rotVecData.x, packet.rotVecData.y, packet.rotVecData.z, packet.rotVecData.w])
                euler = rotation.as_euler(seq='XZY', degrees=True)

                ts = ns_to_s(float(packet.tsNs))
                ts_rel = ts - start_ts

                imu_data.append((
                    packet.gyroData.x, packet.gyroData.y, packet.gyroData.z,
                    packet.accelData.x, packet.accelData.y, packet.accelData.z,
                    *euler,
                    packet.rotVecData.w, packet.rotVecData.x, packet.rotVecData.y, packet.rotVecData.z,
                    packet.tsNs,
                    ts,
                    ts_rel
                ))

                spinner.next()

            spinner.finish()

            self.raw = np.array(imu_data, dtype=IMURecording.DTYPE_RAW).view(
                np.recarray
            )

        log.debug("NeonRecording: Finished parsing IMU packets.")

        num_ts_during_init = self.ts.size - len(self.raw)
        if num_ts_during_init > 0:
            self.ts = self.ts[num_ts_during_init:]
