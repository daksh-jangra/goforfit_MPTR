"""
AI Height Estimator — GoForFit BMI Module
==========================================
Uses MediaPipe PoseLandmarker (v0.10.30+) + OpenCV to estimate
a person's height in real-time from a webcam feed.

SETUP:
    pip install opencv-python mediapipe numpy pandas

CALIBRATION (do this once, first time):
    1. Have a person whose exact height you know stand at 230 cm from camera
    2. Press [C] to enter calibration mode
    3. The app will ask you to type their real height in cm
    4. It computes a SCALE_FACTOR and saves it to calibration.txt
    5. All future readings use this factor — works for everyone

CONTROLS:
    [SPACE]  Start 5-second scan — stands still, then returns final averaged height
    [C]      Calibrate with a known-height person
    [R]      Reset calibration
    [S]      Save all captured readings to CSV
    [Q]      Quit
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
import os
import time
import urllib.request
from datetime import datetime

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
KNOWN_DISTANCE_CM   = 215
CAMERA_FOV_VERTICAL = 50
SMOOTHING_FRAMES    = 15      # more frames = more stable
SCAN_SECONDS        = 5       # how long to collect readings when SPACE is pressed
MIN_VISIBILITY      = 0.55
CALIBRATION_FILE    = "calibration.txt"

MODEL_FILENAME = "pose_landmarker_full.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

BaseOptions        = mp.tasks.BaseOptions
PoseLandmarker     = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOpts = mp.tasks.vision.PoseLandmarkerOptions
# PoseLandmark (index enum) does not exist under mp.tasks.vision in the
# MediaPipe Tasks API (only PoseLandmarker / PoseLandmarksConnections do) —
# referencing it raised: AttributeError: module ‘mediapipe.tasks.python.vision’
# has no attribute ‘PoseLandmark’. It was never used elsewhere in this file
# (landmark indices are hardcoded below as IDX_* constants), so it is
# removed rather than replaced.
VisionRunningMode  = mp.tasks.vision.RunningMode
Connections        = mp.tasks.vision.PoseLandmarksConnections

IDX_NOSE           = 0
IDX_LEFT_SHOULDER  = 11
IDX_RIGHT_SHOULDER = 12
IDX_LEFT_ANKLE     = 27
IDX_RIGHT_ANKLE    = 28
IDX_LEFT_HEEL      = 29
IDX_RIGHT_HEEL     = 30
IDX_LEFT_HIP       = 23
IDX_RIGHT_HIP      = 24


# ─────────────────────────────────────────────────────────
# CALIBRATION  load / save
# ─────────────────────────────────────────────────────────
def load_calibration() -> float:
    """Returns scale factor. 1.0 = no calibration yet."""
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE) as f:
                val = float(f.read().strip())
                if 0.5 < val < 2.0:   # sanity check
                    return val
        except Exception:
            pass
    return 1.0


def save_calibration(factor: float):
    with open(CALIBRATION_FILE, "w") as f:
        f.write(str(round(factor, 6)))


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def download_model(path: str):
    print(f"⬇  Downloading pose model to '{path}' (~7 MB) ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
        print("✅  Model downloaded.")
    except Exception as e:
        print(f"❌  Download failed: {e}")
        print(f"    Manually download from:\n    {MODEL_URL}")
        raise

def get_corrected_distance(landmarks, frame_w: int,
                            base_distance_cm: float,
                            fov_h_deg: float = 70.0) -> float:
    nose = landmarks[IDX_NOSE]
    mid_x = 0.5
    offset_ratio = nose.x - mid_x
    fov_h_rad = math.radians(fov_h_deg)
    angle_rad = offset_ratio * fov_h_rad
    corrected = base_distance_cm / math.cos(angle_rad)
    return corrected


def pixel_height_to_cm(pixel_h: float, frame_h: int,
                        distance_cm: float, fov_deg: float) -> float:
    fov_rad = math.radians(fov_deg)
    return 2 * distance_cm * math.tan(fov_rad / 2) * (pixel_h / frame_h)


def get_height_pixels(landmarks, frame_h: int, frame_w: int):
    """
    Measures pixel span from estimated head-top to lowest foot landmark.
    Uses hip width as a secondary reference for head offset (more stable
    than shoulder width alone, less affected by arm position).
    """
    def lm(idx):
        return landmarks[idx]

    nose = lm(IDX_NOSE)
    if nose.visibility < MIN_VISIBILITY:
        return None

    # Lowest foot landmark
    bottom_y = None
    for idx in [IDX_LEFT_HEEL, IDX_RIGHT_HEEL, IDX_LEFT_ANKLE, IDX_RIGHT_ANKLE]:
        pt = lm(idx)
        if pt.visibility >= MIN_VISIBILITY:
            y = pt.y * frame_h
            if bottom_y is None or y > bottom_y:
                bottom_y = y

    if bottom_y is None:
        return None

    # Head-top offset: average of shoulder-based and hip-based estimates
    # This makes it consistent across different body types
    ls = lm(IDX_LEFT_SHOULDER)
    rs = lm(IDX_RIGHT_SHOULDER)
    lh = lm(IDX_LEFT_HIP)
    rh = lm(IDX_RIGHT_HIP)

    offsets = []
    if ls.visibility >= MIN_VISIBILITY and rs.visibility >= MIN_VISIBILITY:
        shoulder_w_px = abs(ls.x - rs.x) * frame_w
        offsets.append(shoulder_w_px * 0.30)   # tuned: 30% of shoulder width

    if lh.visibility >= MIN_VISIBILITY and rh.visibility >= MIN_VISIBILITY:
        hip_w_px = abs(lh.x - rh.x) * frame_w
        offsets.append(hip_w_px * 0.55)         # hip width is ~55% of head offset

    head_offset = float(np.mean(offsets)) if offsets else 0

    top_y  = (nose.y * frame_h) - head_offset
    height = bottom_y - top_y
    return max(height, 0) if height > 20 else None


def draw_skeleton(frame, landmarks, frame_h, frame_w):
    connections = list(Connections.POSE_LANDMARKS)
    for conn in connections:
        a = landmarks[conn.start]
        b = landmarks[conn.end]
        if a.visibility >= MIN_VISIBILITY and b.visibility >= MIN_VISIBILITY:
            pt1 = (int(a.x * frame_w), int(a.y * frame_h))
            pt2 = (int(b.x * frame_w), int(b.y * frame_h))
            cv2.line(frame, pt1, pt2, (0, 200, 100), 2, cv2.LINE_AA)
    for lm in landmarks:
        if lm.visibility >= MIN_VISIBILITY:
            cx, cy = int(lm.x * frame_w), int(lm.y * frame_h)
            cv2.circle(frame, (cx, cy), 4, (60, 230, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, (0, 100, 180),   1, cv2.LINE_AA)


def draw_hud(frame, smoothed_cm, captured, status_msg, scale_factor,
             calib_mode, calib_input, scan_active, scan_secs_left, final_height):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (10, 15, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "GoForFit  |  AI Height Estimator",
                (12, 26), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 195, 50), 1, cv2.LINE_AA)

    # Calibration badge
    if scale_factor != 1.0:
        badge = f"Calibrated x{scale_factor:.3f}"
        badge_color = (80, 220, 80)
    else:
        badge = "Not calibrated — press [C]"
        badge_color = (60, 160, 255)
    cv2.putText(frame, badge, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, badge_color, 1, cv2.LINE_AA)

    # Height display — show final result if scan just finished, else live reading
    if final_height:
        txt   = f"RESULT: {final_height:.1f} cm   ({final_height/100:.2f} m)"
        color = (50, 255, 200)
    elif smoothed_cm:
        txt   = f"Height: {smoothed_cm:.1f} cm   ({smoothed_cm/100:.2f} m)"
        color = (70, 235, 110)
    else:
        txt   = "Height: align full body in frame..."
        color = (120, 180, 255)
    cv2.putText(frame, txt, (12, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.88, color, 2, cv2.LINE_AA)

    cv2.putText(frame, f"Saved: {len(captured)}",
                (w - 130, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 190, 190), 1, cv2.LINE_AA)

    # Bottom bar
    cv2.rectangle(frame, (0, h - 34), (w, h), (10, 15, 30), -1)
    cv2.putText(frame, "[SPACE] Scan 5s   [C] Calibrate   [R] Reset   [S] Save   [Q] Quit",
                (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

    if status_msg:
        cv2.putText(frame, status_msg,
                    (12, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 220, 255), 2, cv2.LINE_AA)

    # ── Scan countdown overlay ──
    if scan_active:
        # Dim the frame slightly
        dim = frame.copy()
        cv2.rectangle(dim, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(dim, 0.35, frame, 0.65, 0, frame)

        # Pulsing circle countdown
        radius = 80
        cx, cy = w // 2, h // 2
        # background circle
        cv2.circle(frame, (cx, cy), radius, (30, 30, 30), -1)
        cv2.circle(frame, (cx, cy), radius, (80, 80, 80), 2)
        # countdown number
        num_str = str(int(math.ceil(scan_secs_left)))
        (tw, th), _ = cv2.getTextSize(num_str, cv2.FONT_HERSHEY_DUPLEX, 2.5, 3)
        cv2.putText(frame, num_str,
                    (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 220, 60), 3, cv2.LINE_AA)
        # label
        cv2.putText(frame, "SCANNING — STAND STILL",
                    (cx - 180, cy + radius + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

    # Calibration input overlay
    if calib_mode:
        cv2.rectangle(frame, (0, h//2 - 60), (w, h//2 + 60), (10, 10, 40), -1)
        cv2.putText(frame, "CALIBRATION MODE",
                    (w//2 - 160, h//2 - 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 195, 50), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Enter real height in cm: {calib_input}_",
                    (w//2 - 220, h//2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press ENTER to confirm, ESC to cancel",
                    (w//2 - 220, h//2 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA)

    # Height bar
    if smoothed_cm:
        rx, rt, rb = w - 22, 95, h - 38
        rh         = rb - rt
        fill       = int(rh * min(smoothed_cm / 220, 1.0))
        cv2.rectangle(frame, (rx, rt),        (rx + 14, rb),        (45, 45, 45),    -1)
        cv2.rectangle(frame, (rx, rb - fill), (rx + 14, rb),        (70, 210, 110),  -1)
        cv2.rectangle(frame, (rx, rt),        (rx + 14, rb),        (100, 100, 100),  1)
        cv2.putText(frame, f"{smoothed_cm:.0f}", (rx - 12, rb - fill - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 210, 110), 1, cv2.LINE_AA)


def save_csv(records: list, filename: str = None) -> str:
    if not filename:
        filename = f"height_readings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(records).to_csv(filename, index=False)
    return filename


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def _autosave_bmi(session_id, height_cm, weight_kg=None, calibration_used=False):
    """Stages the result on the server for human review/save on screen3 —
    does NOT write to the database directly."""
    try:
        import requests
        requests.post(
            "http://localhost:5000/api/stage/bmi",
            json={
                "session_id":       session_id,
                "height_cm":        height_cm,
                "weight_kg":        weight_kg,
                "calibration_used": calibration_used,
            },
            timeout=2,
        )
        print(f"  [DB] BMI staged for review → height={height_cm} cm  session={session_id}")
    except Exception as e:
        print(f"  [DB] Stage failed (server not running?): {e}")


def main():
    if not os.path.exists(MODEL_FILENAME):
        download_model(MODEL_FILENAME)

    # ── Session ID for DB auto-save (passed via --session-id N or env SESSION_ID) ──
    import sys as _sys, os as _os
    _session_id = None
    for _i, _a in enumerate(_sys.argv[1:], 1):
        if _a == "--session-id" and _i + 1 < len(_sys.argv):
            try: _session_id = int(_sys.argv[_i + 1])
            except ValueError: pass
    if _session_id is None:
        try: _session_id = int(_os.environ.get("SESSION_ID", ""))
        except (ValueError, TypeError): pass

    scale_factor = load_calibration()
    if scale_factor != 1.0:
        print(f"✅  Loaded calibration: scale factor = {scale_factor:.4f}")
    else:
        print("⚠   No calibration found. Press [C] in the window to calibrate.")

    options = PoseLandmarkerOpts(
        base_options                  = BaseOptions(model_asset_path=MODEL_FILENAME),
        running_mode                  = VisionRunningMode.VIDEO,
        num_poses                     = 1,
        min_pose_detection_confidence = 0.70,
        min_pose_presence_confidence  = 0.70,
        min_tracking_confidence       = 0.60,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    height_buffer : list[float] = []
    captured      : list[dict]  = []
    status_msg    = ""
    status_timer  = 0
    frame_idx     = 0
    stream_start_time = time.time()
    last_timestamp_ms  = -1

    # Calibration state
    calib_mode  = False
    calib_input = ""
    calib_raw   = None

    # Scan state
    scan_active      = False
    scan_start_time  = None
    scan_readings    : list[float] = []   # collects ALL readings during 5s scan
    final_height     = None               # locked result shown after scan

    print(f"✅  Running. Stand ~{KNOWN_DISTANCE_CM} cm from camera, full body visible.")
    print("   SPACE=capture | C=calibrate | R=reset calib | S=save CSV | Q=quit\n")

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]
            frame            = cv2.flip(frame, 1)
            frame_idx       += 1

            mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Live webcams (esp. on Windows/DirectShow) report CAP_PROP_POS_MSEC
            # unreliably -- it can jitter or go backwards between frames, which
            # violates MediaPipe VIDEO mode's "strictly increasing timestamp"
            # requirement and crashes detect_for_video(). Use a real wall-clock
            # timer instead, with a guard that force-increments it if two frames
            # would otherwise land on (or before) the same millisecond.
            timestamp_ms = int((time.time() - stream_start_time) * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms
            result       = landmarker.detect_for_video(mp_image, timestamp_ms)

            raw_median      = None
            smoothed_height = None

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                draw_skeleton(frame, landmarks, frame_h, frame_w)

                px_h = get_height_pixels(landmarks, frame_h, frame_w)
                if px_h:
                    # Off-center correction is applied ONCE, through the distance:
                    # get_corrected_distance() already divides the depth by
                    # cos(angle). Previously px_h was ALSO divided by cos(angle),
                    # doubling the correction (~cos^2) so off-center people read
                    # too tall. px_h now passes through uncorrected; centered
                    # people are unaffected (cos(0)=1).
                    corrected_dist = get_corrected_distance(
                        landmarks, frame_w, KNOWN_DISTANCE_CM
                    )
                    raw_cm = pixel_height_to_cm(
                        px_h, frame_h, corrected_dist, CAMERA_FOV_VERTICAL
                    )

                    height_buffer.append(raw_cm)
                    if len(height_buffer) > SMOOTHING_FRAMES:
                        height_buffer.pop(0)

                    raw_median      = float(np.median(height_buffer))
                    smoothed_height = raw_median * scale_factor

                    # collect reading if scan is active
                    if scan_active:
                        scan_readings.append(smoothed_height)

            # ── Scan countdown logic ──
            scan_secs_left = 0
            if scan_active:
                elapsed        = time.time() - scan_start_time
                scan_secs_left = max(0, SCAN_SECONDS - elapsed)

                if scan_secs_left <= 0:
                    # Scan finished — compute final averaged height
                    scan_active = False
                    if scan_readings:
                        final_height = round(float(np.mean(scan_readings)), 1)
                        rec = {
                            "timestamp"    : datetime.now().isoformat(),
                            "height_cm"    : final_height,
                            "height_m"     : round(final_height / 100, 3),
                            "readings_used": len(scan_readings),
                            "distance_cm"  : KNOWN_DISTANCE_CM,
                            "scale_factor" : round(scale_factor, 4),
                        }
                        captured.append(rec)
                        # ── Auto-save to DB ──
                        if _session_id is not None:
                            _autosave_bmi(
                                _session_id, final_height,
                                calibration_used=(scale_factor != 1.0)
                            )
                        status_msg   = f"Done!  Final height: {final_height:.1f} cm  ({len(scan_readings)} readings averaged)"
                        status_timer = 180
                        print(f"  Scan complete: {final_height:.1f} cm  ({len(scan_readings)} frames averaged)")
                    else:
                        status_msg   = "Scan failed — body not detected. Try again."
                        status_timer = 90
                    scan_readings = []

            if status_timer > 0:
                status_timer -= 1
            else:
                status_msg = ""

            draw_hud(frame, smoothed_height, captured, status_msg,
                     scale_factor, calib_mode, calib_input,
                     scan_active, scan_secs_left, final_height)
            cv2.imshow("GoForFit — AI Height Estimator", frame)

            key = cv2.waitKey(1) & 0xFF

            # ── Calibration text input mode ──
            if calib_mode:
                if key == 13:   # ENTER — confirm
                    try:
                        real_h = float(calib_input)
                        if calib_raw and 50 < real_h < 250:
                            scale_factor = real_h / calib_raw
                            save_calibration(scale_factor)
                            status_msg   = f"Calibrated! Factor={scale_factor:.4f}  ({calib_raw:.1f} -> {real_h:.1f} cm)"
                            status_timer = 150
                            print(f"  ✅ Calibrated: raw={calib_raw:.1f} cm  real={real_h:.1f} cm  factor={scale_factor:.4f}")
                        else:
                            status_msg   = "Invalid height or no reading — try again"
                            status_timer = 90
                    except ValueError:
                        status_msg   = "Invalid input — numbers only"
                        status_timer = 90
                    calib_mode  = False
                    calib_input = ""

                elif key == 27:   # ESC — cancel
                    calib_mode  = False
                    calib_input = ""
                    status_msg   = "Calibration cancelled"
                    status_timer = 60

                elif 48 <= key <= 57:   # digits 0–9
                    calib_input += chr(key)

                elif key == 46:   # decimal point
                    if "." not in calib_input:
                        calib_input += "."

                elif key == 8:   # backspace
                    calib_input = calib_input[:-1]

                continue   # skip normal key handling while in calib mode

            # ── Normal key handling ──
            if key == ord(' '):       # Start 5-second scan
                if scan_active:
                    status_msg   = "Scan already running..."
                    status_timer = 40
                elif smoothed_height:
                    scan_active     = True
                    scan_start_time = time.time()
                    scan_readings   = []
                    final_height    = None
                    status_msg      = "Scanning..."
                    status_timer    = 20
                    print("  Scan started — stand still for 5 seconds")
                else:
                    status_msg   = "No pose detected — ensure full body is visible"
                    status_timer = 60

            elif key == ord('c'):     # Enter calibration mode
                if raw_median:
                    calib_raw   = raw_median
                    calib_mode  = True
                    calib_input = ""
                    final_height = None
                    print("  Calibration mode — type the person's real height and press ENTER")
                else:
                    status_msg   = "No pose detected — stand in frame first"
                    status_timer = 60

            elif key == ord('r'):     # Reset calibration
                scale_factor = 1.0
                if os.path.exists(CALIBRATION_FILE):
                    os.remove(CALIBRATION_FILE)
                status_msg   = "Calibration reset"
                status_timer = 90
                print("  Calibration reset.")

            elif key == ord('s'):     # Save CSV
                if captured:
                    fname        = save_csv(captured)
                    status_msg   = f"Saved {len(captured)} records -> {fname}"
                    status_timer = 120
                    print(f"  Saved -> {fname}")
                else:
                    status_msg   = "Nothing captured yet"
                    status_timer = 60

            elif key == ord('q'):     # Quit
                break

    cap.release()
    cv2.destroyAllWindows()

    if captured:
        fname = save_csv(captured)
        print(f"\n  Auto-saved {len(captured)} readings -> {fname}")
    print("Exited.")


if __name__ == "__main__":
    main()