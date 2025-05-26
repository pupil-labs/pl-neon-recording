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
    "Start time (ns since unix epoch)": f"{recording.start_ts}",
    "Start time (datetime)": f"{datetime.fromtimestamp(recording.start_ts / 1e9)}",
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
    recording.eye_state,
    recording.blinks,
    recording.fixations,
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

# Data can be converted to numpy or dataframes, however multi column properties
# like .xy will not be included

print()
print("Gaze data as numpy array:")
gaze_np = recording.gaze.data
print()
print(gaze_np)


print()
print("Gaze data as pandas dataframe:")
gaze_df = recording.gaze.pd
print()
print(gaze_df)


print()
print("Getting data from a stream:")

print()
print("Gaze data", recording.gaze)
# GazeArray([(1741948698620648018, 966.3677 , 439.58817),
#            (1741948698630654018, 965.9669 , 441.60403),
#            (1741948698635648018, 964.2665 , 442.4974 ), ...,
#            (1741948717448190018, 757.85815, 852.34644),
#            (1741948717453190018, 766.53174, 857.3709 ),
#            (1741948717458190018, 730.93604, 851.53723)],
#           dtype=[('ts', '<i8'), ('x', '<f4'), ('y', '<f4')])

print()
print("Gaze ts via prop", recording.gaze.ts)
print("Gaze ts via key", recording.gaze["ts"])
# array([1741948698620648018, 1741948698630654018, 1741948698635648018, ...,
#        1741948717448190018, 1741948717453190018, 1741948717458190018])

print()
print("Gaze x coords via prop", recording.gaze.x)
print("Gaze x coords via key", recording.gaze["x"])
# array([966.3677 , 965.9669 , 964.2665 , ..., 757.85815, 766.53174,
#        730.93604], dtype=float32)

print()
print("Gaze y coords via prop", recording.gaze.y)
print("Gaze y coords via key", recording.gaze["y"])
# array([439.58817, 441.60403, 442.4974 , ..., 852.34644, 857.3709 ,
#        851.53723], dtype=float32)

print()
print("Gaze xy coords", recording.gaze.xy)
print("Gaze xy coords", recording.gaze[["x", "y"]])
# array([[966.3677 , 439.58817],
#        ...,
#        [730.93604, 851.53723]], dtype=float32)

print()
print("Gaze ts and x and y", recording.gaze[["ts", "x", "y"]])
# array([[1.74194870e+18, 9.66367676e+02, 4.39588165e+02],
#        ...,
#        [1.74194872e+18, 7.30936035e+02, 8.51537231e+02]])

print()
print("Sampling data:")

print()
print("Get closest gaze for scene frames")
closest_gaze_to_scene = recording.gaze.sample(recording.scene.ts)
print(closest_gaze_to_scene)
print(
    "closest_gaze_to_scene_times",
    (closest_gaze_to_scene.ts - recording.start_ts) / 1e9,
)


print()
print("Get closest before gaze for scene frames")
closest_gaze_before_scene = recording.gaze.sample(recording.scene.ts, method="backward")
print(closest_gaze_before_scene)
print(
    "closest_gaze_before_scene_times",
    (closest_gaze_before_scene.ts - recording.start_ts) / 1e9,
)


print()
print("Sampled data can be resampled")

print()
print("Closest gaze sampled at 1 fps")
closest_gaze_to_scene_at_one_fps = closest_gaze_before_scene.sample(
    np.arange(closest_gaze_to_scene.ts[0], closest_gaze_to_scene.ts[-1], 1e9 / 1)
)
print(closest_gaze_to_scene_at_one_fps)
print(
    "closest_gaze_to_scene_at_one_fps_times",
    (closest_gaze_to_scene_at_one_fps.ts - recording.start_ts) / 1e9,
)
