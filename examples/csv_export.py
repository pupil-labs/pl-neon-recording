import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import pupil_labs.neon_recording as nr


def unproject_points(points_2d, camera_matrix, distortion_coefs, normalize=False):
    """Undistorts points according to the camera model.

    :param pts_2d, shape: Nx2
    :return: Array of unprojected 3d points, shape: Nx3
    """
    # Convert type to numpy arrays (OpenCV requirements)
    camera_matrix = np.array(camera_matrix)
    distortion_coefs = np.array(distortion_coefs)
    points_2d = np.asarray(points_2d, dtype=np.float32)

    # Add third dimension the way cv2 wants it
    points_2d = points_2d.reshape((-1, 1, 2))

    # Undistort 2d pixel coordinates
    points_2d_undist = cv2.undistortPoints(points_2d, camera_matrix, distortion_coefs)
    # Unproject 2d points into 3d directions; all points. have z=1
    points_3d = cv2.convertPointsToHomogeneous(points_2d_undist)
    points_3d.shape = -1, 3

    if normalize:
        # normalize vector length to 1
        points_3d /= np.linalg.norm(points_3d, axis=1)[:, np.newaxis]

    return points_3d


def cart_to_spherical(points_3d, apply_rad2deg=True):
    points_3d = np.asarray(points_3d)
    # convert cartesian to spherical coordinates
    # source: http://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]
    radius = np.sqrt(x**2 + y**2 + z**2)
    # elevation: vertical direction
    #   positive numbers point up
    #   negative numbers point bottom
    elevation = np.arccos(y / radius) - np.pi / 2
    # azimuth: horizontal direction
    #   positive numbers point right
    #   negative numbers point left
    azimuth = np.pi / 2 - np.arctan2(z, x)

    if apply_rad2deg:
        elevation = np.rad2deg(elevation)
        azimuth = np.rad2deg(azimuth)

    return radius, elevation, azimuth


def find_ranged_index(values, left_boundaries, right_boundaries):
    left_ids = np.searchsorted(left_boundaries, values, side="right") - 1
    right_ids = np.searchsorted(right_boundaries, values, side="right")

    return np.where(left_ids == right_ids, left_ids, -1)


def export_gaze(recording, export_path):
    fixations = recording.fixations[recording.fixations["event_type"] == 1]

    fixation_ids = (
        find_ranged_index(recording.gaze.ts, fixations.start_ts, fixations.end_ts) + 1
    )

    blink_ids = (
        find_ranged_index(
            recording.gaze.ts, recording.blinks.start_ts, recording.blinks.end_ts
        )
        + 1
    )

    spherical_coords = cart_to_spherical(
        unproject_points(
            recording.gaze.xy,
            recording.calibration.scene_camera_matrix,
            recording.calibration.scene_distortion_coefficients,
        )
    )

    gaze = pd.DataFrame({
        "recording id": recording.info["recording_id"],
        "timestamp [ns]": recording.gaze.ts,
        "gaze x [px]": recording.gaze.x,
        "gaze y [px]": recording.gaze.y,
        "worn": recording.worn.worn,
        "fixation id": fixation_ids,
        "blink id": blink_ids,
        "azimuth [deg]": spherical_coords[2],
        "elevation [deg]": spherical_coords[1],
    })

    gaze["fixation id"] = gaze["fixation id"].replace(0, None)
    gaze["blink id"] = gaze["blink id"].replace(0, None)

    export_file = export_path / "gaze.csv"
    gaze.to_csv(export_file, index=False)
    print(f"Wrote {export_file}")


def export_blinks(recording, export_path):
    blinks = pd.DataFrame({
        "recording id": recording.info["recording_id"],
        "blink id": 1 + np.arange(len(recording.blinks)),
        "start timestamp [ns]": recording.blinks.start_ts,
        "end timestamp [ns]": recording.blinks.end_ts,
        "duration [ms]": (recording.blinks.end_ts - recording.blinks.start_ts) / 1e6,
    })
    export_file = export_path / "blinks.csv"
    blinks.to_csv(export_file, index=False)
    print(f"Wrote {export_file}")


def export_fixations(recording, export_path):
    fixations_only = recording.fixations[recording.fixations["event_type"] == 1]

    spherical_coords = cart_to_spherical(
        unproject_points(
            fixations_only.mean_gaze_xy,
            recording.calibration.scene_camera_matrix,
            recording.calibration.scene_distortion_coefficients,
        )
    )

    fixations = pd.DataFrame({
        "recording id": recording.info["recording_id"],
        "fixation id": 1 + np.arange(len(fixations_only)),
        "start timestamp [ns]": fixations_only.start_ts,
        "end timestamp [ns]": fixations_only.end_ts,
        "duration [ms]": (fixations_only.end_ts - fixations_only.start_ts) / 1e6,
        "fixation x [px]": fixations_only.mean_gaze_xy[:, 0],
        "fixation y [px]": fixations_only.mean_gaze_xy[:, 1],
        "azimuth [deg]": spherical_coords[2],
        "elevation [deg]": spherical_coords[1],
    })

    export_file = export_path / "fixations.csv"
    fixations.to_csv(export_file, index=False)
    print(f"Wrote {export_file}")


def export_saccades(recording, export_path):
    saccades_only = recording.fixations[recording.fixations["event_type"] == 0]

    saccades = pd.DataFrame({
        "recording id": recording.info["recording_id"],
        "saccade id": 1 + np.arange(len(saccades_only)),
        "start timestamp [ns]": saccades_only.start_ts,
        "end timestamp [ns]": saccades_only.end_ts,
        "duration [ms]": (saccades_only.end_ts - saccades_only.start_ts) / 1e6,
        "amplitude [px]": saccades_only.amplitude_pixels,
        "amplitude [deg]": saccades_only.amplitude_angle_deg,
        "mean velocity [px/s]": saccades_only.mean_velocity,
        "peak velocity [px/s]": saccades_only.max_velocity,
    })

    export_file = export_path / "saccades.csv"
    saccades.to_csv(export_file, index=False)
    print(f"Wrote {export_file}")


def export_eyestates(recording, export_path):
    es = recording.eye_state
    eyestates = pd.DataFrame({
        "recording id": recording.info["recording_id"],
        "timestamp [ns]": es.ts,
        "pupil diameter left [mm]": es.pupil_diameter_left_mm,
        "pupil diameter right [mm]": es.pupil_diameter_right_mm,
        "eyeball center left x [mm]": es.eyeball_center_left_xyz[:, 0],
        "eyeball center left y [mm]": es.eyeball_center_left_xyz[:, 1],
        "eyeball center left z [mm]": es.eyeball_center_left_xyz[:, 2],
        "eyeball center right x [mm]": es.eyeball_center_right_xyz[:, 0],
        "eyeball center right y [mm]": es.eyeball_center_right_xyz[:, 1],
        "eyeball center right z [mm]": es.eyeball_center_right_xyz[:, 2],
        "optical axis left x": es.optical_axis_left_xyz[:, 0],
        "optical axis left y": es.optical_axis_left_xyz[:, 1],
        "optical axis left z": es.optical_axis_left_xyz[:, 2],
        "optical axis right x": es.optical_axis_right_xyz[:, 0],
        "optical axis right y": es.optical_axis_right_xyz[:, 1],
        "optical axis right z": es.optical_axis_right_xyz[:, 2],
        "eyelid angle top left [rad]": es.eyelid_angle[:, 0],
        "eyelid angle bottom left [rad]": es.eyelid_angle[:, 1],
        "eyelid aperture left [mm]": es.eyelid_aperture_left_right_mm[:, 0],
        "eyelid angle top right [rad]": es.eyelid_angle[:, 2],
        "eyelid angle bottom right [rad]": es.eyelid_angle[:, 3],
        "eyelid aperture right [mm]": es.eyelid_aperture_left_right_mm[:, 1],
    })

    export_file = export_path / "3d_eye_states.csv"
    eyestates.to_csv(export_file, index=False)
    print(f"Wrote {export_file}")


def export_imu(recording, export_path):
    rotations = Rotation.from_quat(recording.imu.quaternion_wxyz, scalar_first=True)
    eulers = rotations.as_euler(seq="yxz", degrees=True)

    imu = pd.DataFrame({
        "recording id": recording.info["recording_id"],
        "timestamp [ns]": recording.imu.ts,
        "gyro x [deg/s]": recording.imu.gyro_xyz[:, 0],
        "gyro y [deg/s]": recording.imu.gyro_xyz[:, 1],
        "gyro z [deg/s]": recording.imu.gyro_xyz[:, 2],
        "acceleration x [g]": recording.imu.accel_xyz[:, 0],
        "acceleration y [g]": recording.imu.accel_xyz[:, 1],
        "acceleration z [g]": recording.imu.accel_xyz[:, 2],
        "roll [deg]": eulers[:, 0],
        "pitch [deg]": eulers[:, 1],
        "yaw [deg]": eulers[:, 2],
        "quaternion w": recording.imu.quaternion_wxyz[:, 0],
        "quaternion x": recording.imu.quaternion_wxyz[:, 1],
        "quaternion y": recording.imu.quaternion_wxyz[:, 2],
        "quaternion z": recording.imu.quaternion_wxyz[:, 3],
    })

    export_file = export_path / "imu.csv"
    imu.to_csv(export_file, index=False)
    print(f"Wrote {export_file}")


def export_events(recording, export_path):
    events = pd.DataFrame({
        "recording id": recording.info["recording_id"],
        "timestamp [ns]": recording.events.ts,
        "name": recording.events.event,
        "type": "recording",
    })

    export_file = export_path / "events.csv"
    events.to_csv(export_file, index=False)
    print(f"Wrote {export_file}")


def export_info(recording, export_path):
    with (export_path / "info.json").open("w") as f:
        json.dump(recording.info, f, indent=4, sort_keys=True)


def export_scene_camera_calibration(recording, export_path):
    distortion = recording.calibration.scene_distortion_coefficients.reshape([1, -1])
    camera_info = {
        "camera_matrix": recording.calibration.scene_camera_matrix.tolist(),
        "distortion_coefficients": distortion.tolist(),
        "serial_number": recording.calibration.serial,
    }
    with (export_path / "scene_camera.json").open("w") as f:
        json.dump(camera_info, f, indent=4, sort_keys=True)


def export_world_timestamps(recording, export_path):
    events = pd.DataFrame({
        "recording id": recording.info["recording_id"],
        "timestamp [ns]": recording.scene.ts,
    })

    export_file = export_path / "world_timestamps.csv"
    events.to_csv(export_file, index=False)
    print(f"Wrote {export_file}")


if __name__ == "__main__":
    func_map = {
        "gaze": export_gaze,
        "blinks": export_blinks,
        "fixations": export_fixations,
        "saccades": export_saccades,
        "eyestates": export_eyestates,
        "imu": export_imu,
        "events": export_events,
        "info": export_info,
        "scene-camera": export_scene_camera_calibration,
        "world-timestamps": export_world_timestamps,
    }

    if len(sys.argv) < 2 or "--help" in sys.argv:
        arg_list = "[--help] [--all] "
        arg_list += " ".join([f"[--{s}]" for s in func_map])
        print(f"Usage: python csv_export.py path/to/recording/folder {arg_list}")
        sys.exit(0)

    recording_path = Path(sys.argv[1])
    recording = nr.open(recording_path)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    safe_timestamp = timestamp.replace(":", "-").replace(" ", "_")

    export_path = recording_path / "exports" / safe_timestamp
    export_path.mkdir(parents=True, exist_ok=True)

    if "--all" in sys.argv or len(sys.argv) == 2:
        sys.argv += [f"--{k}" for k in func_map]

    for stream_name, export_func in func_map.items():
        if f"--{stream_name}" in sys.argv:
            export_func(recording, export_path)
