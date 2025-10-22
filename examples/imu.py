import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# Workaround for https://github.com/opencv/opencv/issues/21952
cv2.imshow("cv/av bug", np.zeros(1))
cv2.destroyAllWindows()

import pupil_labs.neon_recording as nr  # noqa: E402

if len(sys.argv) < 2:
    print("Usage:")
    print("python imu.py path/to/recording/folder")

# Open a recording
recording = nr.open(sys.argv[1])

# Sample the IMU data at 60Hz
fps = 60
z = recording.gaze[:10]
timestamps = np.arange(
    recording.imu.time[0], recording.imu.time[-1], 1e9 / fps, dtype=np.int64
)
imu_data = recording.imu.sample(timestamps)

# Use scipy to convert the quaternions to euler angles
quaternions = np.array([s.rotation for s in imu_data])
rotations = Rotation.from_quat(quaternions).as_euler(seq="yxz", degrees=True) % 360

# Combine the timestamps and eulers
rotations_with_time = np.column_stack((timestamps, rotations))
timestamped_eulers = np.array(
    [tuple(row) for row in rotations_with_time],
    dtype=[
        ("time", np.int64),
        ("roll", np.float64),
        ("pitch", np.float64),
        ("yaw", np.float64),
    ],
)

# Display the angles
frame_size = 512
colors = {"pitch": (0, 0, 255), "yaw": (0, 255, 0), "roll": (255, 0, 0)}

for row in tqdm(timestamped_eulers):
    # Create a blank image
    frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)

    # Define the center and radius of the circles
    center = [frame_size // 2] * 2
    radius = frame.shape[0] // 3

    # Calculate the end points for the angles
    for field, color in colors.items():
        pitch_end = (
            int(center[0] + radius * np.cos(np.deg2rad(row[field]))),
            int(center[1] - radius * np.sin(np.deg2rad(row[field]))),
        )
        cv2.line(frame, center, pitch_end, color, 2)

        # Write the angle values on the image
        cv2.putText(
            frame,
            f"{field}: {row[field]:.2f}",
            (10, 30 + list(colors.keys()).index(field) * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    # Display the image
    cv2.imshow("IMU Angles", frame)
    if cv2.waitKey(1000 // fps) == 27:
        break

cv2.destroyAllWindows()
