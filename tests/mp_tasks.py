"""MediaPipe Tasks-API helpers shared by the engine scripts.

Google removed the legacy `mediapipe.solutions` API (Hands, Pose,
drawing_utils, drawing_styles) in mediapipe 0.10.30, which is why this
project was pinned to `mediapipe<0.10.30` — and, because there are no
sub-0.10.30 wheels for Python 3.13, pinned to Python <= 3.12 as a knock-on.
This module is what lets both pins go.

Two things lived in `solutions` and had to be replaced:

  DETECTION  Hands()/Pose() with their stateful .process(rgb). The Tasks
             equivalents are HandLandmarker/PoseLandmarker in VIDEO mode,
             which track across frames the same way but need a monotonically
             increasing timestamp per detector — see VideoClock.

  DRAWING    drawing_utils.draw_landmarks(), which took a protobuf landmark
             list. The Tasks API hands back plain objects, so the drawing
             here is straight cv2 lines and circles over the connection
             pairs the Tasks API itself publishes. parcurlup.py has done it
             this way since its own port; this generalises that.

Landmark objects from the Tasks API expose .x/.y/.z normalised to the frame
plus .visibility, so call sites that read `lm.visibility` keep working.
There is no PoseLandmark enum under tasks — use the integer indices below.
"""

import os
import urllib.request

import cv2

from mediapipe import Image, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Pose landmark indices. mp.solutions.pose.PoseLandmark is gone and the Tasks
# API ships no replacement enum, so the few we actually use are named here
# rather than left as magic numbers at the call sites.
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

MODEL_URLS = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}

_HERE = os.path.dirname(os.path.abspath(__file__))


def model_path(name, folder=None):
    """Absolute path to a .task model, downloading it once if absent.

    Resolves next to this file by default, so it works no matter which
    directory the server or a test script was started from — the scripts
    used to look the models up relative to cwd and fail when launched by
    the web app.
    """
    folder = folder or _HERE
    dst = os.path.join(folder, name)
    if not os.path.exists(dst):
        url = MODEL_URLS.get(name)
        if url is None:
            raise FileNotFoundError(f"No download URL known for model {name!r}")
        print(f"[DOWNLOAD] {name} …", end=" ", flush=True)
        urllib.request.urlretrieve(url, dst)
        print(f"done ({os.path.getsize(dst) // 1024} KB)")
    return dst


# ── detector builders ─────────────────────────────────────────────────────

def hand_landmarker(num_hands=2, detection_conf=0.70, presence_conf=0.50,
                    tracking_conf=0.60, model="hand_landmarker.task"):
    """VIDEO mode: keeps identity across frames, like solutions.Hands did."""
    return mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path(model)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=presence_conf,
            min_tracking_confidence=tracking_conf,
        )
    )


def pose_landmarker(num_poses=1, detection_conf=0.5, presence_conf=0.5,
                    tracking_conf=0.5, model="pose_landmarker_lite.task"):
    return mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path(model)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=detection_conf,
            min_pose_presence_confidence=presence_conf,
            min_tracking_confidence=tracking_conf,
        )
    )


def to_mp_image(bgr_frame):
    """OpenCV BGR array -> the SRGB mp.Image the Tasks API expects."""
    return Image(image_format=ImageFormat.SRGB,
                 data=cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))


class VideoClock:
    """Monotonic millisecond timestamps for one detector.

    VIDEO mode rejects a timestamp that does not advance, and each detector
    keeps its own clock — shuttle_run runs one per lane per camera, so a
    single shared counter would have them fighting each other. Counting
    frames rather than reading the wall clock keeps it monotonic even if two
    frames arrive within the same millisecond.
    """

    def __init__(self, frame_interval_ms=33):
        self._ms = 0
        self._step = frame_interval_ms

    def tick(self):
        self._ms += self._step
        return self._ms


# ── drawing (replaces solutions.drawing_utils / drawing_styles) ────────────

def _connection_pairs(connection_list, fallback):
    """The Tasks API publishes connections as objects with .start/.end.
    Fall back to literals if a future version moves them again — drawing is
    cosmetic and must never be the thing that breaks a test run."""
    try:
        return [(c.start, c.end) for c in connection_list]
    except Exception:
        return fallback


HAND_CONNECTIONS = _connection_pairs(
    getattr(mp_vision.HandLandmarksConnections, "HAND_CONNECTIONS", []),
    [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
     (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
     (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)],
)

POSE_CONNECTIONS = _connection_pairs(
    getattr(mp_vision.PoseLandmarksConnections, "POSE_LANDMARKS", []),
    [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
     (11, 23), (12, 24), (23, 24),
     (23, 25), (24, 26), (25, 27), (26, 28)],
)

_LANDMARK_COLOR = (80, 160, 255)
_CONNECTION_COLOR = (255, 255, 255)


def draw_landmarks(frame, landmarks, connections, point_color=_LANDMARK_COLOR,
                   line_color=_CONNECTION_COLOR, min_visibility=0.0):
    """One person's/hand's landmarks onto a BGR frame, in place."""
    h, w = frame.shape[:2]

    def visible(lm):
        return getattr(lm, "visibility", 1.0) is None or getattr(lm, "visibility", 1.0) >= min_visibility

    def px(lm):
        return int(lm.x * w), int(lm.y * h)

    for a, b in connections:
        if a < len(landmarks) and b < len(landmarks) and visible(landmarks[a]) and visible(landmarks[b]):
            cv2.line(frame, px(landmarks[a]), px(landmarks[b]), line_color, 1, cv2.LINE_AA)
    for lm in landmarks:
        if visible(lm):
            cv2.circle(frame, px(lm), 3, point_color, -1, cv2.LINE_AA)


def draw_hand_skeleton(frame, hand_landmarks_list):
    for landmarks in hand_landmarks_list or []:
        draw_landmarks(frame, landmarks, HAND_CONNECTIONS)


def draw_pose_skeleton(frame, pose_result, min_visibility=0.4):
    for landmarks in getattr(pose_result, "pose_landmarks", None) or []:
        draw_landmarks(frame, landmarks, POSE_CONNECTIONS,
                       point_color=(180, 80, 80), line_color=(80, 80, 180),
                       min_visibility=min_visibility)
