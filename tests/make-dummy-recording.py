import math
import numpy as np
import cv2
from pathlib import Path
import json
import time
import pupil_labs.video as plv
from fractions import Fraction


def putText(image,
            text,
            org,
            font_face,
            font_scale,
            color,
            thickness=1,
            line_type=cv2.LINE_8):
    image = cv2.putText(image, text, org, font_face, font_scale, (0, 0, 0),
                        math.ceil(thickness * 5))
    image = cv2.putText(image, text, org, font_face, font_scale, color,
                        thickness)

    return image


start_time = time.time() * 1e9

recording_path = Path(__file__).parent / "dummy"
recording_path.mkdir(exist_ok=True)

with (recording_path / "info.json").open("wt") as info_file:
    info = {
        "recording_id": "00000000-0000-0000-0000-000000000000",
        "workspace_id": "00000000-0000-0000-0000-000000000000",
        "data_format_version": "2.2",
        "app_version": "2.8.3-stage",
        "start_time": start_time,
        "duration": 71389000000,
        "android_device_id": "0000000000000000",
        "android_device_name": "Neon Companion",
        "android_device_model": "NE2217",
        "module_serial_number": "000000",
        "wearer_id": "00000000-0000-0000-0000-000000000000",
        "template_data": {
            "id": "00000000-0000-0000-0000-000000000000",
            "version": "2023-04-19T09:19:46.201495Z",
            "recording_name": "dummy-recording",
            "data": {}
        },
        "gaze_offset": [0.0, 0.0],
        "calib_version": 1,
        "pipeline_version": "2.6.0",
        "os_version": "14",
        "frame_id": "1",
        "frame_name": "Just act natural",
        "firmware_version": [24, 8],
        "gaze_frequency": 200,
        "wearer_ied": 63.0
    }
    json.dump(info, info_file, indent=4)

with (recording_path / "wearer.json").open("wt") as wearer_file:
    json.dump({
        "uuid": "00000000-0000-0000-0000-000000000000",
        "name": "dummy"
    }, wearer_file)

print("Scene video...")
for section_idx in range(3):
    video_part_file = recording_path / f"Neon Scene Camera v1 ps{section_idx+1}.mp4"
    output_container = plv.open(video_part_file, mode="w")
    out_video_stream = output_container.add_stream("libx264", rate=30)
    out_video_stream.width = 1600
    out_video_stream.height = 1200
    out_video_stream.time_base = Fraction(1, 30)

    # 5 seconds of video
    for frame_idx in range(150):
        frame_pixels = np.zeros([1200, 1600, 3], dtype='uint8')
        frame_pixels[:, :, section_idx] = frame_idx
        frame_pixels = putText(frame_pixels,
                               f"Part {section_idx} - Frame {frame_idx}",
                               (20, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0,
                               (255, 255, 255), 2)
        output_frame = plv.VideoFrame.from_ndarray(frame_pixels,
                                                   format='bgr24')
        output_frame.pts = frame_idx

        for packet in out_video_stream.encode(output_frame):
            output_container.mux(packet)

    for packet in out_video_stream.encode():
        output_container.mux(packet)

    output_container.close()

    section_start_time = start_time + (section_idx * 10 * 1e9)
    end_time = section_start_time + 150 / 30 * 1e9
    output_timestamps = np.arange(section_start_time,
                                  end_time,
                                  1 / 30 * 1e9,
                                  dtype="<u8")
    output_timestamps.tofile(video_part_file.with_suffix(".time"))

print("Eye frames...")
# eye frames
rng = np.random.default_rng()
for section_idx in range(1):
    video_part_file = recording_path / f"Neon Sensor Module v1 ps{section_idx+1}.mp4"
    output_container = plv.open(video_part_file, mode="w")
    out_video_stream = output_container.add_stream("libx264", rate=200)
    out_video_stream.width = 384
    out_video_stream.height = 192
    out_video_stream.time_base = Fraction(1, 200)

    for frame_idx in range(200 * 30):
        frame_pixels = rng.integers(255, size=[192, 384], dtype='uint8')
        output_frame = plv.VideoFrame.from_ndarray(frame_pixels, format='gray')
        output_frame.pts = frame_idx

        for packet in out_video_stream.encode(output_frame):
            output_container.mux(packet)

    for packet in out_video_stream.encode():
        output_container.mux(packet)

    output_container.close()

    section_start_time = start_time
    end_time = start_time + 30 * 1e9
    output_timestamps = np.arange(start_time,
                                  end_time,
                                  1 / 200 * 1e9,
                                  dtype="<u8")
    output_timestamps.tofile(video_part_file.with_suffix(".time"))

# gaze data

gaze_part_file = recording_path / f"gaze ps1.raw"
gaze_fps = 5
point_count = gaze_fps * 30
points = np.arange(0, 1200, 1200 / point_count, dtype="<f4")
points = np.column_stack([points, points])
points.tofile(gaze_part_file)

end_time = start_time + 30 * 1e9
gaze_times = np.arange(start_time, end_time, 1e9 / gaze_fps, dtype="<u8")
gaze_times.tofile(gaze_part_file.with_suffix(".time"))

# events
events_file = recording_path / "event.txt"
with events_file.open("wt") as output_file:
    print("recording.begin\nrecording.end\n", file=output_file)
event_times = np.array([start_time, end_time])
event_times.tofile(events_file.with_suffix(".time"))
