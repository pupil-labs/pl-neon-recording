import json
import pathlib

from . import structlog
from .calib import Calibration, parse_calib_bin
from .stream.gaze_stream import GazeStream
from .stream.event_stream import EventStream
from .stream.imu import IMUStream
from .stream.eye_state_stream import EyeStateStream
from .stream.av_stream.video_stream import VideoStream
from .stream.av_stream.audio_stream import AudioStream

log = structlog.get_logger(__name__)


class NeonRecording:
    def __init__(self, rec_dir_in: pathlib.Path | str):
        self._rec_dir = pathlib.Path(rec_dir_in).resolve()
        if not self._rec_dir.exists() or not self._rec_dir.is_dir():
            raise FileNotFoundError(f"Directory not found or not valid: {self._rec_dir}")

        log.info(f"NeonRecording: Loading recording from {rec_dir_in}")

        log.info("NeonRecording: Loading recording info")
        with open(self._rec_dir / "info.json") as f:
            self.info = json.load(f)

        self.start_ts_ns = self.info["start_time"]
        self.start_ts = self.start_ts_ns * 1e-9

        log.info("NeonRecording: Loading wearer")
        self.wearer = {"uuid": "", "name": ""}
        with open(self._rec_dir / "wearer.json") as f:
            wearer_data = json.load(f)

        self.wearer["uuid"] = wearer_data["uuid"]
        self.wearer["name"] = wearer_data["name"]


        log.info("NeonRecording: Loading calibration data")
        self._calib = parse_calib_bin(self._rec_dir)

        self.calib_version = str(self._calib["version"])
        self.serial = int(self._calib["serial"][0])
        self.scene_camera_calibration = Calibration(
            self._calib["scene_camera_matrix"],
            self._calib["scene_distortion_coefficients"],
            self._calib["scene_extrinsics_affine_matrix"],
        )
        self.right_eye_camera_calibration = Calibration(
            self._calib["right_camera_matrix"],
            self._calib["right_distortion_coefficients"],
            self._calib["right_extrinsics_affine_matrix"],
        )
        self.left_eye_camera_calibration = Calibration(
            self._calib["left_camera_matrix"],
            self._calib["left_distortion_coefficients"],
            self._calib["left_extrinsics_affine_matrix"],
        )

        log.info("NeonRecording: Loading data streams")
        self.streams = {}

        log.info("NeonRecording: Finished loading recording.")

    # TODO: save for the end of development
    def check(self):
        pass

    @property
    def gaze(self) -> GazeStream:
        if "gaze" not in self.streams:
            self.streams["gaze"] = GazeStream(self)

        return self.streams["gaze"]

    @property
    def imu(self) -> IMUStream:
        if "imu" not in self.streams:
            self.streams["imu"] = IMUStream(self)

        return self.streams["imu"]

    @property
    def eye_state(self) -> EyeStateStream:
        if "eye_state" not in self.streams:
            self.streams["eye_state"] = EyeStateStream(self)

        return self.streams["eye_state"]

    @property
    def scene(self) -> SceneVideoStream:
        if "scene" not in self.streams:
            self.streams["scene"] = VideoStream("scene", "Neon Scene Camera v1", self)

        return self.streams["scene"]

    @property
    def eye(self) -> VideoStream:
        if "eye" not in self.streams:
            self.streams["eye"] = VideoStream("eye", "Neon Sensor Module v1", self)

        return self.streams["eye"]

    @property
    def events(self) -> EventStream:
        if "event" not in self.streams:
            self.streams["event"] = EventStream(self)

        return self.streams["events"]

    @property
    def audio(self) -> AudioStream:
        if "scene" not in self.streams:
            self.streams["audio"] = AudioStream(self)

        return self.streams["audio"]



def load(rec_dir_in: pathlib.Path | str) -> NeonRecording:
    return NeonRecording(rec_dir_in)
