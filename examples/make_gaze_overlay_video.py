import sys
from fractions import Fraction

import cv2
import numpy as np
import pupil_labs.neon_recording as nr
import pupil_labs.video as plv

rec = nr.load(sys.argv[1])

gaze = rec.gaze
eye = rec.eye
scene = rec.scene
scene_video = scene.video_stream
scene_audio = scene.audio_stream
imu = rec.imu


def transparent_rect(img, x, y, w, h):
    sub_img = img[y : y + h, x : x + w]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 85
    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
    img[y : y + h, x : x + w] = res


def overlay_image(img, img_overlay, x, y):
    """Overlay `img_overlay` onto `img` at (x, y)."""

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    img_crop[:] = img_overlay_crop


def convert_neon_pts_to_video_pts(neon_pts, neon_time_base, video_time_base):
    return int(float(neon_pts * neon_time_base) / video_time_base)


fps = 65535
container = plv.open("video.mp4", mode="w")

out_video_stream = container.add_stream("mpeg4", rate=fps)
out_video_stream.width = scene.width
out_video_stream.height = scene.height
out_video_stream.pix_fmt = "yuv420p"

out_audio_stream = container.add_stream("aac", rate=scene_audio.sample_rate)

neon_time_base = scene.data[0].time_base
video_time_base = Fraction(1, fps)

avg_neon_pts_size = int(np.mean(np.diff([f.pts for f in scene.data if f is not None])))
avg_video_pts_size = convert_neon_pts_to_video_pts(
    avg_neon_pts_size, neon_time_base, video_time_base
)

start_ts = rec.unique_events["recording.begin"]
end_ts = rec.unique_events["recording.end"]

avg_frame_dur = np.mean(np.diff(scene.ts))
pre_ts = np.arange(start_ts, scene.ts[0] - avg_frame_dur, avg_frame_dur)
post_ts = np.arange(scene.ts[-1] + avg_frame_dur, end_ts, avg_frame_dur)

my_ts = np.concatenate((pre_ts, scene.ts, post_ts))

fields = [
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "pitch",
    "yaw",
    "roll",
    "accel_x",
    "accel_y",
    "accel_z",
]
colors = [
    (208, 203, 228),
    (135, 157, 115),
    (179, 133, 124),
    (101, 118, 223),
    (189, 201, 138),
    (235, 167, 124),
    (93, 197, 128),
    (188, 181, 0),
    (24, 50, 170),
]
imu_maxes = {}
for field in fields:
    imu_maxes[field] = np.max(np.abs(imu[field]))

ts_rel_max = np.max(imu.ts_rel)

# gyro_data = dict.fromkeys(fields, [])
gyro_data = {}
for field in fields:
    gyro_data[field] = []

pts_offset = 0
video_pts = 0
reached_video_start = False
combined_data = zip(
    gaze.sample(my_ts),
    eye.sample(my_ts),
    scene_video.sample(my_ts),
    imu.sample(my_ts),
    scene_audio.sample(my_ts),
)
for gaze_datum, eye_frame, scene_frame, imu_datum, audio_sample in combined_data:
    scene_image = (
        scene_frame.cv2
        if scene_frame is not None
        else np.ones((scene.height, scene.width, 3), dtype="uint8") * 128  # gray frames
    )
    eye_image = (
        eye_frame.cv2
        if eye_frame is not None
        else np.zeros((eye.height, eye.width, 3), dtype="uint8")  # black frames
    )

    overlay_image(scene_image, eye_image, 0, 0)
    if gaze_datum:
        cv2.circle(
            scene_image, (int(gaze_datum.x), int(gaze_datum.y)), 50, (0, 0, 255), 10
        )

    border_cols = [(245, 201, 176), (225, 181, 156), (225, 181, 156)]
    transparent_rect(scene_image, 0, 950, scene.width, scene.height - 950)
    cv2.line(scene_image, (0, 950), (scene.width, 950), border_cols[0], 2)
    cv2.line(scene_image, (0, 950 + 80), (scene.width, 950 + 80), border_cols[1], 2)
    cv2.line(scene_image, (0, 950 + 160), (scene.width, 950 + 160), border_cols[2], 2)
    if imu_datum:
        sep = 0
        for i, field in enumerate(imu_maxes):
            if i > 0 and i % 3 == 0:
                sep += 80

            datum_mx = imu_maxes[field]
            gyro_data[field].append(
                [
                    scene.width * imu_datum.ts_rel / ts_rel_max,
                    -1.0 * imu_datum[field] / datum_mx * 20 + 1000 + sep,
                ]
            )

            cv2.polylines(
                scene_image,
                np.array([gyro_data[field]], dtype=np.int32),
                isClosed=False,
                color=colors[i],
                thickness=2,
            )

    frame = plv.VideoFrame.from_ndarray(scene_image, format="bgr24")
    if scene_frame is not None:
        reached_video_start = True
        video_pts = convert_neon_pts_to_video_pts(
            scene_frame.pts, neon_time_base, video_time_base
        )
    elif reached_video_start and scene_frame is None:
        video_pts += avg_video_pts_size

    frame.pts = pts_offset + video_pts
    frame.time_base = video_time_base
    for packet in out_video_stream.encode(frame):
        container.mux(packet)

    if scene_frame is not None:
        cv2.imshow("frame", scene_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if not reached_video_start and scene_frame is None:
        pts_offset += avg_video_pts_size

try:
    # Flush out_video_stream
    for packet in out_video_stream.encode():
        container.mux(packet)
finally:
    # Close the file
    container.close()
