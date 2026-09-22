#!/usr/bin/env python3
"""
=============================================================
  Alternate Hand Wall Toss Test — CV Counter
=============================================================
  Automatically counts catches in the standard alternate-hand
  wall-toss coordination test using a webcam.

  Test protocol:
    1. Stand ~1 m from a wall, camera to the side or slightly
       behind you so it can see both hands and the ball.
    2. Toss the ball: RIGHT hand → wall → LEFT hand (catch)
    3. Toss back:     LEFT hand → wall → RIGHT hand (catch)
    4. Repeat — your score is the total alternating catches
       made in 30 seconds.

  Requirements:
    pip install opencv-python mediapipe numpy

  Controls:
    S      — Start / restart the 30-second test
    R      — Reset to zero
    C      — Enter colour-calibration mode,
             then LEFT-CLICK the ball in the frame
    1–5    — Switch ball colour preset
               1=Orange  2=Yellow  3=Red  4=Green  5=Blue
    Q      — Quit
=============================================================
"""

import os
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

# Tasks API (works on tasks-only MediaPipe builds; the legacy
# mp.solutions.hands used before is absent from newer wheels).
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision
from mediapipe.tasks.python.vision import HandLandmarksConnections as _HandConns


def _draw_hand_cv(frame, landmarks, connections):
    """Draw a hand skeleton with OpenCV — replaces mp.solutions.drawing_utils.
    `landmarks` is a Tasks-API list of normalized landmarks (.x/.y);
    `connections` is an iterable of objects with .start/.end indices."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    n = len(pts)
    for c in connections:
        s, e = c.start, c.end
        if 0 <= s < n and 0 <= e < n:
            cv2.line(frame, pts[s], pts[e], (255, 255, 255), 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 3, (80, 160, 255), -1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  (edit these to suit your environment)
# ──────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX         = 0       # Webcam index (try 1 or 2 if 0 fails)
FRAME_W              = 1280    # Capture resolution (camera may cap this)
FRAME_H              = 720
TEST_DURATION        = 30      # Seconds per test

# Minimum contour area (px²) for a blob to be treated as the ball
BALL_MIN_AREA        = 180   # lowered: motion blur fragments the colour blob
# Circularity score threshold (0 = any shape, 1 = perfect circle)
BALL_MIN_CIRCULARITY = 0.28   # lowered: a fast ball motion-blurs into an oval
                              # streak; 0.45 rejected it. Colour already excludes
                              # hands/arms, so a looser roundness check is safe.
# Pixel radius around hand palm centre that counts as "holding"
HAND_PROXIMITY_PX    = 130
# Ball must exceed this speed (px / frame) to register as a throw
THROW_SPEED_PX_FR    = 6.0
# Seconds that must elapse between two counted tosses (debounce)
TOSS_COOLDOWN_SEC    = 0.25
# After this many consecutive missing frames the ball is considered lost
MAX_BALL_MISS_FR     = 12
BALL_PREDICT_FR      = 4    # bridge up to N blurred/dropped frames by extrapolating


# ──────────────────────────────────────────────────────────────────────────────
#  BALL COLOUR PRESETS  (HSV lower, upper)
# ──────────────────────────────────────────────────────────────────────────────
BALL_PRESETS = {
    1: ("Orange", np.array([  5, 100, 100]), np.array([ 25, 255, 255])),
    2: ("Yellow", np.array([ 22,  70,  70]), np.array([ 45, 255, 255])),  # widened to cover optic-yellow TENNIS balls (yellow-green)
    3: ("Red",    np.array([  0, 150, 100]), np.array([ 10, 255, 255])),
    4: ("Green",  np.array([ 36, 100, 100]), np.array([ 86, 255, 255])),
    5: ("Blue",   np.array([ 94,  80,  70]), np.array([126, 255, 255])),
}


# ──────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (BGR — used for drawing)
# ──────────────────────────────────────────────────────────────────────────────
C_CYAN   = (255, 220,   0)
C_GREEN  = ( 50, 210,  50)
C_RED    = ( 50,  50, 210)
C_WHITE  = (255, 255, 255)
C_YELLOW = ( 50, 210, 210)
C_GREY   = (160, 160, 160)
C_DARK   = ( 20,  20,  20)
C_ORANGE = ( 30, 140, 255)


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN CLASS
# ──────────────────────────────────────────────────────────────────────────────
def _autosave_wall_toss(session_id, catches, ball_colour=None):
    """Stages the result on the server for human review/save on screen3 —
    does NOT write to the database directly."""
    try:
        import requests
        requests.post(
            "http://localhost:8080/api/stage/wall_toss",
            json={
                "session_id":  session_id,
                "catches":     catches,
                "duration_sec": 30,
                "ball_colour": ball_colour,
            },
            timeout=2,
        ).raise_for_status()
        print(f"  [DB] wall_toss staged for review → catches={catches}  session={session_id}")
    except Exception as e:
        print(f"  [DB] Stage failed: {e}")


def _other_hand(hand: str) -> str:
    """Return the opposite hand label."""
    return "Left" if hand == "Right" else "Right"


class WallTossCounter:
    """
    Five-state machine:

        ready ──► right_holding ──► right_tossed ──► left_holding
                  ◄── (miss)                               │
                                                           ▼
          right_holding ◄── left_tossed ◄──────────── left_holding
                  (miss) ──►

    Each completed right_tossed → left_holding  transition = +1 toss.
    Each completed left_tossed  → right_holding transition = +1 toss.
    """

    STATES = ("ready",
              "right_holding", "right_tossed",
              "left_holding",  "left_tossed")

    def __init__(self, mirror=True):
        # Selfie-mirror the preview. Standalone run() flips its own frames, so
        # this only matters to the browser engine (WallTossEngine), which mirrors
        # the frame itself before handing it to _detect_ball/_process_hands and
        # passes mirror here so both sides agree on orientation.
        self.mirror = mirror

        # ── MediaPipe Hands (Tasks API) ───────────────────────────────────
        # Ported off legacy mp.solutions.hands (absent from tasks-only builds
        # like 0.10.35). Same 2-hand VIDEO tracking; landmark indices
        # (0=wrist, 9=middle-MCP) and 'Left'/'Right' labels are unchanged.
        _model = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "hand_landmarker.task")
        _opts = _mp_vision.HandLandmarkerOptions(
            base_options=_mp_python.BaseOptions(model_asset_path=_model),
            running_mode=_mp_vision.RunningMode.VIDEO,
            num_hands=2,
            # Lowered from 0.70/0.60: a fast-moving, motion-blurred, or
            # edge-of-frame catching hand scores low confidence and was being
            # dropped (~40% detection on the moving hand), so its catches never
            # registered. Looser thresholds detect the hands far more reliably;
            # the 130px catch-proximity check tolerates the slightly noisier
            # landmark positions this brings.
            min_hand_detection_confidence=0.40,
            min_hand_presence_confidence=0.40,
            min_tracking_confidence=0.40,
        )
        self._detector = _mp_vision.HandLandmarker.create_from_options(_opts)
        self._hand_conns = _HandConns.HAND_CONNECTIONS
        self._ts_ms = 0

        # ── Ball state ────────────────────────────────────────────────────
        _, lo, hi = BALL_PRESETS[1]          # default = orange
        self.ball_lo:   np.ndarray = lo.copy()
        self.ball_hi:   np.ndarray = hi.copy()
        self._preset_name          = BALL_PRESETS[1][0]

        self.ball_center: tuple | None = None
        self.ball_radius: int          = 0
        self.ball_path   = deque(maxlen=45)   # trail for visualisation
        self.ball_hist   = deque(maxlen=15)   # recent positions for speed
        self.ball_miss   = 0                  # consecutive detection failures

        # ── Hand state ────────────────────────────────────────────────────
        # keys: 'Left' | 'Right'  →  {'center': (cx, cy), 'landmarks': ...}
        self.hands: dict = {}

        # ── Test / scoring state ──────────────────────────────────────────
        self.state        = "ready"
        self.toss_count   = 0
        self.last_toss_t  = 0.0
        self.start_t: float | None = None
        self.active       = False
        self.done         = False
        self.final_score  = 0
        # Airborne-latch counting: _hold = hand that last held the ball;
        # _airborne latches True once a genuinely fast (thrown) ball is seen, so
        # a toss still counts even if the exact in-flight frame was dropped —
        # while a slow hand-to-hand pass (never fast) never latches and is not
        # counted.
        self._hold: str | None = None
        self._airborne         = False

        # ── UI helpers ────────────────────────────────────────────────────
        self.hint_hand: str | None = None  # which hand to use next
        self.flash         = 0             # frames left to show green flash
        self.status_msg    = "Press [S] to start the test"
        self._calib_mode   = False
        self._frame_buf: np.ndarray | None = None   # for mouse callback

        # ── resolution-aware detection thresholds ─────────────────────────
        # HAND_PROXIMITY_PX and THROW_SPEED_PX_FR are pixel measurements tuned
        # for the desktop capture width (FRAME_W = 1280). At a smaller width the
        # same physical motion spans proportionally fewer pixels, so the browser
        # engine — which runs at ~960 px — would otherwise need the ball closer
        # and thrown "faster" (in px) than intended. Keep instance copies the
        # engine can rescale for the real frame width via set_frame_scale();
        # standalone run() never calls it, so it keeps the exact 1280 tuning.
        self.hand_proximity_px = float(HAND_PROXIMITY_PX)
        self.throw_speed_px_fr = float(THROW_SPEED_PX_FR)
        self._frame_scale      = 1.0

    def set_frame_scale(self, frame_w: int):
        """Rescale the pixel-based thresholds for the actual frame width so the
        *physical* sensitivity matches the desktop tuning at any resolution.
        Scales linearly by frame_w / FRAME_W. Idempotent — cheap to call every
        frame; only recomputes when the width actually changes."""
        if not frame_w or frame_w <= 0:
            return
        scale = frame_w / FRAME_W
        if scale == self._frame_scale:
            return
        self._frame_scale      = scale
        self.hand_proximity_px = HAND_PROXIMITY_PX * scale
        self.throw_speed_px_fr = THROW_SPEED_PX_FR * scale

    # ─────────────────────────────────────────────────────────────────────
    #  Ball detection helpers
    # ─────────────────────────────────────────────────────────────────────

    def _detect_ball(self, bgr: np.ndarray) -> tuple[tuple | None, int]:
        """HSV colour filter + contour analysis.  Returns (center, radius)."""
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.ball_lo, self.ball_hi)

        k    = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        # Bridge motion-blur gaps so a streaked, fragmented fast ball still forms
        # one connected blob (raises detection rate for fast tosses).
        mask = cv2.dilate(mask, k, iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, 0

        best = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area < BALL_MIN_AREA:
            return None, 0

        # Reject non-round blobs (e.g. hands, arms)
        peri  = cv2.arcLength(best, True)
        circ  = (4 * np.pi * area / peri ** 2) if peri > 0 else 0
        if circ < BALL_MIN_CIRCULARITY:
            return None, 0

        (x, y), r = cv2.minEnclosingCircle(best)
        return (int(x), int(y)), max(1, int(r))

    def _ball_speed(self) -> float:
        """Average ball speed in px/frame over recent history."""
        pts = [p for p in self.ball_hist if p is not None]
        if len(pts) < 3:
            return 0.0
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        return np.hypot(dx, dy) / len(pts)

    def calibrate_ball(self, bgr: np.ndarray, click: tuple, r: int = 18):
        """Derive HSV range from a clicked pixel region."""
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        x, y = int(click[0]), int(click[1])
        roi  = hsv[max(0, y-r):y+r, max(0, x-r):x+r]
        if roi.size == 0:
            return
        mn  = roi.mean(axis=(0, 1))
        sd  = roi.std(axis=(0, 1))
        pad = np.array([12, 35, 35], dtype=float)
        self.ball_lo = np.clip(mn - 2*sd - pad,
                               [0,  50,  50], [179, 255, 255]).astype(np.uint8)
        self.ball_hi = np.clip(mn + 2*sd + pad,
                               [0,  50,  50], [179, 255, 255]).astype(np.uint8)
        self._preset_name = "Custom"
        print(f"[Calibrate] HSV centre={mn.astype(int)}  "
              f"lo={self.ball_lo}  hi={self.ball_hi}")

    def set_preset(self, key: int):
        """Switch to a named colour preset."""
        if key in BALL_PRESETS:
            name, lo, hi = BALL_PRESETS[key]
            self.ball_lo      = lo.copy()
            self.ball_hi      = hi.copy()
            self._preset_name = name
            print(f"[Preset] Ball colour → {name}")

    # ─────────────────────────────────────────────────────────────────────
    #  Hand detection helpers
    # ─────────────────────────────────────────────────────────────────────

    def _process_hands(self, rgb: np.ndarray, shape: tuple):
        """Run MediaPipe, populate self.hands, return mp result."""
        self._ts_ms += 33   # VIDEO mode needs increasing timestamps (~30fps)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=np.ascontiguousarray(rgb))
        res = self._detector.detect_for_video(mp_img, self._ts_ms)
        self.hands = {}

        if res.hand_landmarks and res.handedness:
            h, w = shape[:2]
            for lms, side in zip(res.hand_landmarks, res.handedness):
                label = side[0].category_name          # 'Left' or 'Right'
                # The frame is selfie-mirrored before detection, which flips
                # MediaPipe's handedness guess (it assumes a non-mirrored image).
                # Swap it back so the on-screen L/R matches the person's actual
                # hand — the student's right hand reads "Right".
                if self.mirror:
                    label = "Right" if label == "Left" else "Left"
                # Palm centre = midpoint of WRIST (0) and MIDDLE MCP (9)
                wr = lms[0]
                mc = lms[9]
                cx = int((wr.x + mc.x) / 2 * w)
                cy = int((wr.y + mc.y) / 2 * h)
                self.hands[label] = {"center": (cx, cy), "landmarks": lms}
        return res

    def _near(self, label: str) -> bool:
        """True when the ball is within HAND_PROXIMITY_PX of the named hand."""
        if self.ball_center is None or label not in self.hands:
            return False
        bx, by = self.ball_center
        hx, hy = self.hands[label]["center"]
        return float(np.hypot(bx - hx, by - hy)) < self.hand_proximity_px

    def _predict_ball(self):
        """Extrapolate the next ball position from its recent velocity, so a
        motion-blurred frame where the ball isn't detected doesn't lose track of
        a fast toss. Returns a predicted (x, y) or None if there's no motion."""
        pts = [p for p in self.ball_hist if p is not None]
        if len(pts) < 2:
            return None
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        vx, vy = x2 - x1, y2 - y1
        if abs(vx) < 1 and abs(vy) < 1:      # near-stationary → nothing to extrapolate
            return None
        return (int(x2 + vx), int(y2 + vy))

    def ingest_ball(self, bc, br):
        """Update ball state from one detection result. On a miss, bridge short
        gaps (up to BALL_PREDICT_FR frames) with a velocity prediction so fast,
        blurred tosses keep tracking instead of dropping out."""
        if bc is not None:
            self.ball_center = bc
            self.ball_radius = br
            self.ball_miss   = 0
            self.ball_path.append(bc)
            self.ball_hist.append(bc)
            return
        self.ball_miss += 1
        pred = self._predict_ball() if self.ball_miss <= BALL_PREDICT_FR else None
        if pred is not None:
            self.ball_center = pred
            self.ball_path.append(pred)
            self.ball_hist.append(pred)
        else:
            self.ball_hist.append(None)
            if self.ball_miss > MAX_BALL_MISS_FR:
                self.ball_center = None

    def _nearest_hand(self):
        """Return the hand ('Right'/'Left') the ball is closest to within
        proximity, or None if the ball isn't near either hand."""
        if self.ball_center is None:
            return None
        bx, by = self.ball_center
        best, best_d = None, None
        for label in ("Right", "Left"):
            if label in self.hands:
                hx, hy = self.hands[label]["center"]
                d = float(np.hypot(bx - hx, by - hy))
                if d < self.hand_proximity_px and (best_d is None or d < best_d):
                    best, best_d = label, d
        return best

    # ─────────────────────────────────────────────────────────────────────
    #  State machine
    # ─────────────────────────────────────────────────────────────────────

    def _update_state(self):
        """Advance the toss-detection state machine by one frame.

        Airborne-latch model (tolerant of dropped in-flight frames): track which
        hand last held the ball (_hold) and latch _airborne the moment the ball
        is seen genuinely fast/away from the hands. A toss is scored when the ball
        reaches the OTHER hand while _airborne is latched. A slow hand-to-hand
        pass never reaches throw speed, so it never latches and never counts."""
        if not self.active:
            return

        now  = time.time()
        spd  = self._ball_speed()
        near = self._nearest_hand()

        # Latch airborne evidence: any fast ball movement (a real throw) counts,
        # even if it happens while the ball is momentarily near a hand.
        if self._hold is not None and self.ball_center is not None \
                and spd >= self.throw_speed_px_fr:
            self._airborne = True

        if self._hold is None:
            # Wait for a hand to take the ball first.
            if near is not None:
                self._hold     = near
                self._airborne = False
                self.hint_hand = _other_hand(near)
                self.status_msg = f"Toss & catch with the {_other_hand(near).upper()} hand"
        else:
            other = _other_hand(self._hold)
            self.hint_hand = other
            if near == self._hold:
                # Ball back in the holding hand — any pending throw didn't land.
                self._airborne = False
            elif near == other:
                # Ball reached the OTHER hand.
                if self._airborne and now - self.last_toss_t > TOSS_COOLDOWN_SEC:
                    self._register_toss(other)     # ✓ real alternating toss
                    self._hold     = other
                    self._airborne = False
                elif not self._airborne:
                    self.status_msg = "Toss it to the wall — don't just pass it!"
            else:
                # near is None: ball not by either hand — in flight.
                if self.ball_center is not None and spd >= self.throw_speed_px_fr:
                    self._airborne = True

        # Keep self.state as a readable label for the HUD / API.
        if self._hold is None:
            self.state = "ready"
        elif self._airborne:
            self.state = f"{self._hold.lower()}_tossed"
        else:
            self.state = f"{self._hold.lower()}_holding"

    def _register_toss(self, caught_by: str):
        """Record one successful alternating catch."""
        self.toss_count  += 1
        self.last_toss_t  = time.time()
        self.hint_hand    = caught_by
        self.flash        = 15     # frames of green flash
        self.status_msg   = f"✓ Caught by {caught_by}!  Total: {self.toss_count}"
        print(f"  Toss #{self.toss_count:3d}  caught by {caught_by}")

    # ─────────────────────────────────────────────────────────────────────
    #  Test control
    # ─────────────────────────────────────────────────────────────────────

    def start(self):
        self.toss_count  = 0
        self.state       = "ready"
        self.start_t     = time.time()
        self.last_toss_t = 0.0
        self.active      = True
        self.done        = False
        self.final_score = 0
        self._hold       = None
        self._airborne   = False
        self.ball_path.clear()
        self.ball_hist.clear()
        self.status_msg  = "Test running — start tossing!"
        self.hint_hand   = None
        print(f"\n[Start] {TEST_DURATION}s test begins …")

    def reset(self):
        self.toss_count  = 0
        self.state       = "ready"
        self.start_t     = None
        self.active      = False
        self.done        = False
        self.final_score = 0
        self._hold       = None
        self._airborne   = False
        self.ball_path.clear()
        self.ball_hist.clear()
        self.status_msg  = "Press [S] to start the test"
        self.hint_hand   = None
        self.flash       = 0

    # ─────────────────────────────────────────────────────────────────────
    #  Drawing helpers
    # ─────────────────────────────────────────────────────────────────────

    def _draw(self, frame: np.ndarray, mp_res):
        h, w = frame.shape[:2]

        # 1. MediaPipe hand skeleton (OpenCV drawing; Tasks API result)
        if mp_res.hand_landmarks:
            for lms in mp_res.hand_landmarks:
                _draw_hand_cv(frame, lms, self._hand_conns)

        # 2. Ball trajectory trail (fades towards older positions)
        pts = [p for p in self.ball_path if p is not None]
        for i in range(1, len(pts)):
            a   = i / len(pts)
            col = (int(50 + 180*a), int(160*a), int(255*(1-a)))
            cv2.line(frame, pts[i-1], pts[i], col, max(1, int(3*a)))

        # 3. Ball marker
        if self.ball_center:
            cv2.circle(frame, self.ball_center, self.ball_radius + 6, C_YELLOW, 2)
            cv2.circle(frame, self.ball_center, 5, C_ORANGE, -1)
            # Speed text near ball
            spd_txt = f"{self._ball_speed():.1f}"
            cv2.putText(frame, spd_txt,
                        (self.ball_center[0] + self.ball_radius + 8,
                         self.ball_center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_YELLOW, 1)

        # 4. Hand circles with labels
        for label, d in self.hands.items():
            cx, cy  = d["center"]
            col     = C_GREEN if label == "Right" else C_RED
            is_hint = (label == self.hint_hand and self.active)

            if is_hint:
                # Pulsing highlight ring for the active hand
                cv2.circle(frame, (cx, cy), 46, C_YELLOW, 3)

            cv2.circle(frame, (cx, cy), 28, col,    -1)
            cv2.circle(frame, (cx, cy), 28, C_WHITE,  2)
            cv2.putText(frame, label[0], (cx-9, cy+7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, C_WHITE, 2)

            # Green ring when ball is in proximity
            if self._near(label):
                cv2.circle(frame, (cx, cy), 46, C_GREEN, 2)

            # Label below hand circle
            cv2.putText(frame, label, (cx - 22, cy + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1)

        # 5. Green flash on successful catch
        if self.flash > 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 180, 0), -1)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            self.flash -= 1

        # 6. ── Info panel (top-left) ─────────────────────────────────────
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (445, 248), C_DARK, -1)
        cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

        cv2.putText(frame, "ALTERNATE HAND WALL TOSS", (12, 34),
                    cv2.FONT_HERSHEY_DUPLEX, 0.82, C_CYAN, 2)

        # Score
        score_col = C_GREEN if self.toss_count > 0 else C_GREY
        cv2.putText(frame, f"Tosses: {self.toss_count}", (12, 102),
                    cv2.FONT_HERSHEY_DUPLEX, 1.65, score_col, 3)

        # Timer / result
        if self.active and self.start_t is not None:
            rem = max(0.0, TEST_DURATION - (time.time() - self.start_t))
            if rem <= 0 and not self.done:
                self.active      = False
                self.done        = True
                self.final_score = self.toss_count
                self.status_msg  = f"Test complete!  Score = {self.final_score}"
                print(f"\n[Done] Score: {self.final_score} catches in {TEST_DURATION}s")

            t_col = C_GREEN if rem > 10 else C_RED
            cv2.putText(frame, f"Time: {rem:5.1f}s", (12, 154),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, t_col, 2)

            # Progress bar
            bar_end = 15 + int(415 * (1 - rem / TEST_DURATION))
            cv2.rectangle(frame, (15, 170), (430, 185), (60, 60, 60), -1)
            cv2.rectangle(frame, (15, 170), (bar_end, 185), t_col,        -1)

        elif self.done:
            cv2.putText(frame, f"FINAL SCORE: {self.final_score}", (12, 154),
                        cv2.FONT_HERSHEY_DUPLEX, 0.95, C_YELLOW, 2)

        cv2.putText(frame, self.status_msg[:58], (12, 225),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 110), 1)

        # 7. ── HUD (top-right) ───────────────────────────────────────────
        hud_x = w - 270
        ov2 = frame.copy()
        cv2.rectangle(ov2, (hud_x-8, 8), (w-8, 98), C_DARK, -1)
        cv2.addWeighted(ov2, 0.68, frame, 0.32, 0, frame)

        cv2.putText(frame, f"State : {self.state}", (hud_x, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 255), 1)
        cv2.putText(frame, f"Speed : {self._ball_speed():.1f} px/fr", (hud_x, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 255, 180), 1)
        cv2.putText(frame, f"Ball  : {self._preset_name}", (hud_x, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_YELLOW, 1)

        # 8. Next-hand arrow (centre-right or centre-left)
        if self.hint_hand and self.active:
            ay = h // 2 + 80
            if self.hint_hand == "Right":
                cv2.arrowedLine(frame, (w-120, ay), (w-30, ay),
                                C_GREEN, 3, tipLength=0.35)
                cv2.putText(frame, "R", (w-142, ay+8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_GREEN, 2)
            else:
                cv2.arrowedLine(frame, (120, ay), (30, ay),
                                C_RED, 3, tipLength=0.35)
                cv2.putText(frame, "L", (128, ay+8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_RED, 2)

        # 9. Calibration banner
        if self._calib_mode:
            banner = "CALIBRATE MODE — Click on the ball"
            bw = cv2.getTextSize(banner, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0][0]
            cv2.putText(frame, banner, ((w - bw)//2, h//2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

        # 10. Footer controls
        cv2.putText(
            frame,
            "[S] Start   [R] Reset   [C] Calibrate   "
            "[1-5] Ball colour   [Q] Quit",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_GREY, 1,
        )

    # ─────────────────────────────────────────────────────────────────────
    #  Main loop
    # ─────────────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}. "
                               "Try changing CAMERA_INDEX at the top of the file.")

        win = "Alternate Hand Wall Toss Test"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, FRAME_W, FRAME_H)

        # Mouse callback for ball-colour calibration
        def _on_mouse(event, x, y, *_):
            if (event == cv2.EVENT_LBUTTONDOWN
                    and self._calib_mode
                    and self._frame_buf is not None):
                self.calibrate_ball(self._frame_buf, (x, y))
                self._calib_mode = False

        cv2.setMouseCallback(win, _on_mouse)
        print(__doc__)

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("[Warning] Empty frame — retrying …")
                continue

            # Mirror so movement feels natural
            frame = cv2.flip(frame, 1)
            self._frame_buf = frame.copy()   # snapshot for mouse callback

            # ── Ball detection ─────────────────────────────────────────────
            bc, br = self._detect_ball(frame)
            self.ingest_ball(bc, br)

            # ── Hand detection ─────────────────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            mp_res = self._process_hands(rgb, frame.shape)

            # ── State machine ──────────────────────────────────────────────
            self._update_state()

            # ── Draw ───────────────────────────────────────────────────────
            self._draw(frame, mp_res)
            cv2.imshow(win, frame)

            # ── Keyboard input ─────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if   key == ord("q"):
                break
            elif key == ord("s") and not self.active:
                self.start()
            elif key == ord("r"):
                self.reset()
            elif key == ord("c"):
                self._calib_mode = not self._calib_mode
            elif key in (ord("1"), ord("2"), ord("3"),
                         ord("4"), ord("5")):
                self.set_preset(key - ord("0"))

        # ── Cleanup ────────────────────────────────────────────────────────
        cap.release()
        cv2.destroyAllWindows()
        self._detector.close()

        if self.done:
            print(f"\n{'='*46}")
            print(f"  Final score : {self.final_score} catches in {TEST_DURATION}s")
            print(f"{'='*46}")


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    WallTossCounter().run()