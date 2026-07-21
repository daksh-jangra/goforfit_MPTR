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

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time


# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  (edit these to suit your environment)
# ──────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX         = 0       # Webcam index (try 1 or 2 if 0 fails)
FRAME_W              = 1280    # Capture resolution (camera may cap this)
FRAME_H              = 720
TEST_DURATION        = 30      # Seconds per test

# Minimum contour area (px²) for a blob to be treated as the ball
BALL_MIN_AREA        = 300
# Circularity score threshold (0 = any shape, 1 = perfect circle)
BALL_MIN_CIRCULARITY = 0.45
# Pixel radius around hand palm centre that counts as "holding"
HAND_PROXIMITY_PX    = 130
# Ball must exceed this speed (px / frame) to register as a throw
THROW_SPEED_PX_FR    = 6.0
# Seconds that must elapse between two counted tosses (debounce)
TOSS_COOLDOWN_SEC    = 0.35
# After this many consecutive missing frames the ball is considered lost
MAX_BALL_MISS_FR     = 12


# ──────────────────────────────────────────────────────────────────────────────
#  BALL COLOUR PRESETS  (HSV lower, upper)
# ──────────────────────────────────────────────────────────────────────────────
BALL_PRESETS = {
    1: ("Orange", np.array([  5, 100, 100]), np.array([ 25, 255, 255])),
    2: ("Yellow", np.array([ 20, 100, 100]), np.array([ 35, 255, 255])),
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
            "http://localhost:5000/api/stage/wall_toss",
            json={
                "session_id":  session_id,
                "catches":     catches,
                "duration_sec": 30,
                "ball_colour": ball_colour,
            },
            timeout=2,
        )
        print(f"  [DB] wall_toss staged for review → catches={catches}  session={session_id}")
    except Exception as e:
        print(f"  [DB] Stage failed: {e}")


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

    def __init__(self):
        # ── MediaPipe Hands ───────────────────────────────────────────────
        self._mp_h  = mp.solutions.hands
        self._mp_dr = mp.solutions.drawing_utils
        self._mp_ds = mp.solutions.drawing_styles
        self._detector = self._mp_h.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.70,
            min_tracking_confidence=0.60,
        )

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

        # ── UI helpers ────────────────────────────────────────────────────
        self.hint_hand: str | None = None  # which hand to use next
        self.flash         = 0             # frames left to show green flash
        self.status_msg    = "Press [S] to start the test"
        self._calib_mode   = False
        self._frame_buf: np.ndarray | None = None   # for mouse callback

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
        res = self._detector.process(rgb)
        self.hands = {}

        if res.multi_hand_landmarks and res.multi_handedness:
            h, w = shape[:2]
            for lms, side in zip(res.multi_hand_landmarks,
                                  res.multi_handedness):
                label = side.classification[0].label   # 'Left' or 'Right'
                # Palm centre = midpoint of WRIST (0) and MIDDLE MCP (9)
                wr = lms.landmark[0]
                mc = lms.landmark[9]
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
        return float(np.hypot(bx - hx, by - hy)) < HAND_PROXIMITY_PX

    # ─────────────────────────────────────────────────────────────────────
    #  State machine
    # ─────────────────────────────────────────────────────────────────────

    def _update_state(self):
        """Advance the toss-detection state machine by one frame."""
        if not self.active:
            return

        now  = time.time()
        spd  = self._ball_speed()
        R    = self._near("Right")
        L    = self._near("Left")

        if self.state == "ready":
            # Accept whichever hand picks up the ball first
            if R:
                self.state     = "right_holding"
                self.hint_hand = "Right"
            elif L:
                self.state     = "left_holding"
                self.hint_hand = "Left"

        elif self.state == "right_holding":
            self.hint_hand = "Right"
            if not R and spd >= THROW_SPEED_PX_FR:
                self.state      = "right_tossed"
                self.status_msg = "Tossed! → Catch with LEFT hand"

        elif self.state == "right_tossed":
            self.hint_hand = "Left"
            if L and now - self.last_toss_t > TOSS_COOLDOWN_SEC:
                # ✓ Successful alternating catch
                self._register_toss("Left")
                self.state = "left_holding"
            elif R:
                # Ball returned to same hand — no score
                self.state      = "right_holding"
                self.status_msg = "Catch with the OTHER hand!"

        elif self.state == "left_holding":
            self.hint_hand = "Left"
            if not L and spd >= THROW_SPEED_PX_FR:
                self.state      = "left_tossed"
                self.status_msg = "Tossed! → Catch with RIGHT hand"

        elif self.state == "left_tossed":
            self.hint_hand = "Right"
            if R and now - self.last_toss_t > TOSS_COOLDOWN_SEC:
                # ✓ Successful alternating catch
                self._register_toss("Right")
                self.state = "right_holding"
            elif L:
                # Ball returned to same hand — no score
                self.state      = "left_holding"
                self.status_msg = "Catch with the OTHER hand!"

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

        # 1. MediaPipe hand skeleton
        if mp_res.multi_hand_landmarks:
            for lms in mp_res.multi_hand_landmarks:
                self._mp_dr.draw_landmarks(
                    frame, lms,
                    self._mp_h.HAND_CONNECTIONS,
                    self._mp_ds.get_default_hand_landmarks_style(),
                    self._mp_ds.get_default_hand_connections_style(),
                )

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
            if bc is not None:
                self.ball_center = bc
                self.ball_radius = br
                self.ball_miss   = 0
                self.ball_path.append(bc)
                self.ball_hist.append(bc)
            else:
                self.ball_miss += 1
                self.ball_hist.append(None)
                if self.ball_miss > MAX_BALL_MISS_FR:
                    self.ball_center = None

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