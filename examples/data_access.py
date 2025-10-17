import sys
from collections.abc import Mapping
from datetime import datetime

import numpy as np

import pupil_labs.neon_recording as nr
from pupil_labs.neon_recording.timeseries.timeseries import Timeseries

if len(sys.argv) < 2:
    print("Usage:")
    print("python basic_usage.py path/to/recording/folder")

# Open a recording
recording = nr.open(sys.argv[1])


def pretty_format(mapping: Mapping):
    output = []
    pad = "    "
    keys = mapping.keys()
    n = max(len(key) for key in keys)
    for k, v in mapping.items():
        v_repr_lines = str(v).splitlines()
        output.append(f"{pad}{k:>{n}}: {v_repr_lines[0]}")
        if len(v_repr_lines) > 1:
            output.extend(f"{pad + '  '}{n * ' '}{line}" for line in v_repr_lines[1:])
    return "\n".join(output)


print("Basic Recording Info:")
print_data = {
    "Recording ID": recording.id,
    "Start time (ns since unix epoch)": f"{recording.start_time}",
    "Start time (datetime)": f"{datetime.fromtimestamp(recording.start_time / 1e9)}",
    "Duration (nanoseconds)": f"{recording.duration}",
    "Duration (seconds)": f"{recording.duration / 1e9}",
    "Wearer": f"{recording.wearer['name']} ({recording.wearer['uuid']})",
    "Device serial": recording.device_serial,
    "App version": recording.info["app_version"],
    "Data format": recording.info["data_format_version"],
    "Gaze Offset": recording.info["gaze_offset"],
}
print(pretty_format(print_data))

streams: list[Timeseries] = [
    recording.gaze,
    recording.imu,
    recording.eyeball,
    recording.pupil,
    recording.eyelid,
    recording.blinks,
    recording.fixations,
    recording.saccades,
    recording.worn,
    recording.eye,
    recording.scene,
    recording.audio,
]
print()
print("Recording streams:")
print(
    pretty_format({
        f"{stream.name} ({len(stream)} samples)": "\n" + pretty_format(stream[0])
        for stream in streams
    })
)

print()
print("Getting data from a stream:")

print()
print("Gaze points", recording.gaze.point)
# GazeArray([(1741948698620648018, 966.3677 , 439.58817),
#            (1741948698630654018, 965.9669 , 441.60403),
#            (1741948698635648018, 964.2665 , 442.4974 ), ...,
#            (1741948717448190018, 757.85815, 852.34644),
#            (1741948717453190018, 766.53174, 857.3709 ),
#            (1741948717458190018, 730.93604, 851.53723)],
#           dtype=[('ts', '<i8'), ('x', '<f4'), ('y', '<f4')])

print()
print("Gaze timestamps", recording.gaze.time)
# array([1741948698620648018, 1741948698630654018, 1741948698635648018, ...,
#        1741948717448190018, 1741948717453190018, 1741948717458190018])


# All stream data can also be accesses as structured numpy arrays and pandas dataframes.

print()
print("Gaze data as a structured numpy array:")
gaze_np = recording.gaze.data
print()
print(gaze_np)


print()
print("Gaze data as pandas dataframe:")
gaze_df = recording.gaze.pd
print()
print(gaze_df)

print()
print("Sampling data:")

print()
print("Get closest gaze for scene frames")
closest_gaze_to_scene = recording.gaze.sample(recording.scene.time)
print(closest_gaze_to_scene)
print(
    "closest_gaze_to_scene_times",
    (closest_gaze_to_scene.time - recording.start_time) / 1e9,
)

print()
print("Sampled data can be resampled")

print()
print("Closest gaze sampled at 1 fps")
closest_gaze_to_scene_at_one_fps = closest_gaze_to_scene.sample(
    np.arange(
        closest_gaze_to_scene.time[0],
        closest_gaze_to_scene.time[-1],
        1e9 / 1,
        dtype=np.int64,
    ),
)
print(closest_gaze_to_scene_at_one_fps)
print(
    "closest_gaze_to_scene_at_one_fps_times",
    (closest_gaze_to_scene_at_one_fps.time - recording.start_time) / 1e9,
)
