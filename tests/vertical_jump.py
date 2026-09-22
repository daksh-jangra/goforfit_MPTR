import cv2 as cv
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import argparse
import requests

STAGE_URL = "http://localhost:8080/api/stage/vertical_jump"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CALIBRATION_FILE = os.path.join(BASE_DIR, "calibration_vertical_jump.txt")
...
base_options = python.BaseOptions(
    model_asset_path=os.path.join(BASE_DIR, "pose_landmarker_full.task")
)

def stage_result(session_id, attempts_cm, standing_reach_cm):
    if not attempts_cm:
        return
    payload = {
        "session_id": session_id,
        "jump_cm": round(max(attempts_cm), 2),
        "average_jump_cm": round(sum(attempts_cm) / len(attempts_cm), 2),
        "standing_reach_cm": standing_reach_cm,
        "attempts": [round(a, 2) for a in attempts_cm],
    }
    try:
        requests.post(STAGE_URL, json=payload, timeout=3).raise_for_status()
    except Exception as exc:
        print(f"[WARN] Could not stage result: {exc}")


CAMERA_SOURCE = 0
ATTEMPTS_REQUIRED = 5
BASELINE_FRAMES = 30
JUMP_RESET_TOLERANCE = 12
DEFAULT_PIXEL_TO_CM = 0.35
CALIBRATION_FILE = "calibration_vertical_jump.txt"
HAND_EXTENSION_RATIO = 0.75
VISIBILITY_THRESHOLD = 0.55


options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True,
    num_poses=1,
)

model = vision.PoseLandmarker.create_from_options(options)

NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def landmark_to_point(landmark, width, height):
    return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)


def get_body_points(landmarks, width, height):
    left_hip = landmark_to_point(
        landmarks[LEFT_HIP], width, height
    )
    right_hip = landmark_to_point(
        landmarks[RIGHT_HIP], width, height
    )
    left_ankle = landmark_to_point(
        landmarks[LEFT_ANKLE], width, height
    )
    right_ankle = landmark_to_point(
        landmarks[RIGHT_ANKLE], width, height
    )

    nose = landmark_to_point(
        landmarks[NOSE], width, height
    )
    left_eye = landmark_to_point(
        landmarks[LEFT_EYE], width, height
    )
    right_eye = landmark_to_point(
        landmarks[RIGHT_EYE], width, height
    )
    left_ear = landmark_to_point(
        landmarks[LEFT_EAR], width, height
    )
    right_ear = landmark_to_point(
        landmarks[RIGHT_EAR], width, height
    )

    hip_center = (left_hip + right_hip) / 2.0
    ankle_center = (left_ankle + right_ankle) / 2.0
    head_y = min(
        nose[1],
        left_eye[1],
        right_eye[1],
        left_ear[1],
        right_ear[1],
    )

    return hip_center, ankle_center, nose, head_y


def estimate_hand_tip(landmarks, wrist_idx, elbow_idx, width, height):
    wrist = landmarks[wrist_idx]
    elbow = landmarks[elbow_idx]

    if wrist.visibility < VISIBILITY_THRESHOLD or elbow.visibility < VISIBILITY_THRESHOLD:
        return None

    wrist_pt = landmark_to_point(wrist, width, height)
    elbow_pt = landmark_to_point(elbow, width, height)
    tip_pt = wrist_pt + (wrist_pt - elbow_pt) * HAND_EXTENSION_RATIO
    return tip_pt


def get_reach_points(landmarks, width, height):
    left_tip = estimate_hand_tip(landmarks, LEFT_WRIST, LEFT_ELBOW, width, height)
    right_tip = estimate_hand_tip(landmarks, RIGHT_WRIST, RIGHT_ELBOW, width, height)

    candidates = [tip for tip in [left_tip, right_tip] if tip is not None]
    if not candidates:
        return None, None, None

    best_tip = min(candidates, key=lambda pt: pt[1])
    return best_tip, left_tip, right_tip


def draw_pose(frame, landmarks, width, height):
    connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
        (24, 26), (26, 28), (27, 31), (28, 32),
    ]

    for start_idx, end_idx in connections:
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        start_point = (int(start.x * width), int(start.y * height))
        end_point = (int(end.x * width), int(end.y * height))
        cv.line(frame, start_point, end_point, (120, 255, 120), 2)

    key_points = [
        NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE,
    ]

    for idx in key_points:
        lm = landmarks[idx]
        if lm.visibility >= VISIBILITY_THRESHOLD:
            point = (int(lm.x * width), int(lm.y * height))
            cv.circle(frame, point, 4, (255, 240, 120), -1)

    left_tip = estimate_hand_tip(landmarks, LEFT_WRIST, LEFT_ELBOW, width, height)
    right_tip = estimate_hand_tip(landmarks, RIGHT_WRIST, RIGHT_ELBOW, width, height)

    for tip, wrist_idx in [(left_tip, LEFT_WRIST), (right_tip, RIGHT_WRIST)]:
        if tip is not None:
            wrist = landmarks[wrist_idx]
            wrist_pt = (int(wrist.x * width), int(wrist.y * height))
            tip_pt = (int(tip[0]), int(tip[1]))
            cv.line(frame, wrist_pt, tip_pt, (255, 190, 0), 2)
            cv.circle(frame, tip_pt, 6, (0, 220, 255), -1)


def draw_status(frame, lines):
    for index, text in enumerate(lines):
        cv.putText(
            frame,
            text,
            (20, 40 + index * 35),
            cv.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )


def draw_height_shadows(frame, baseline_feet_y, baseline_reach_y, attempt_shadows, pixel_to_cm):
    width = frame.shape[1]

    if baseline_feet_y is not None:
        cv.line(
            frame,
            (width - 260, int(baseline_feet_y)),
            (width - 20, int(baseline_feet_y)),
            (0, 215, 255),
            2,
        )
        cv.putText(
            frame,
            "Feet baseline",
            (width - 255, int(baseline_feet_y) - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 215, 255),
            2,
        )

    if baseline_reach_y is not None:
        cv.line(
            frame,
            (width - 260, int(baseline_reach_y)),
            (width - 20, int(baseline_reach_y)),
            (255, 180, 0),
            2,
        )
        cv.putText(
            frame,
            "Standing reach",
            (width - 255, int(baseline_reach_y) - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 180, 0),
            2,
        )

    for attempt in attempt_shadows:
        reach_y = int(attempt["peak_reach_y"])
        label = f"A{attempt['attempt']}: reach {attempt['reach_cm']:.1f} cm | jump {attempt['jump_cm']:.1f} cm"
        cv.line(
            frame,
            (width - 260, reach_y),
            (width - 20, reach_y),
            attempt["color"],
            3,
        )
        cv.putText(
            frame,
            label,
            (width - 255, reach_y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            attempt["color"],
            2,
        )
        if baseline_feet_y is not None:
            cv.line(
                frame,
                (width - 40, int(baseline_feet_y)),
                (width - 40, reach_y),
                attempt["color"],
                2,
            )


def draw_calibration_overlay(frame, calib_input, pending_jump_px):
    height, width = frame.shape[:2]
    cv.rectangle(frame, (0, height // 2 - 70), (width, height // 2 + 70), (20, 20, 40), -1)
    cv.putText(
        frame,
        "CALIBRATION MODE",
        (width // 2 - 180, height // 2 - 25),
        cv.FONT_HERSHEY_DUPLEX,
        0.9,
        (255, 195, 50),
        2,
    )
    cv.putText(
        frame,
        f"Measured standing reach span: {pending_jump_px:.1f} px",
        (width // 2 - 180, height // 2 + 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
    )
    cv.putText(
        frame,
        f"Enter standing reach height from floor (cm): {calib_input}_",
        (width // 2 - 250, height // 2 + 42),
        cv.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )
    cv.putText(
        frame,
        "Press ENTER to save, ESC to cancel",
        (width // 2 - 180, height // 2 + 68),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (170, 170, 170),
        1,
    )


def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, "r", encoding="utf-8") as file:
                value = float(file.read().strip())
            if 0.01 < value < 5.0:
                return value
        except Exception:
            pass
    return DEFAULT_PIXEL_TO_CM


def save_calibration(pixel_to_cm):
    with open(CALIBRATION_FILE, "w", encoding="utf-8") as file:
        file.write(str(round(pixel_to_cm, 6)))


def main(session_id = None):
    cap = cv.VideoCapture(CAMERA_SOURCE)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Unable to open camera source.")
        return

    pixel_to_cm = load_calibration()
    standing_reach_cm = None
    baseline_feet_y = None
    baseline_reach_y = None
    baseline_feet_samples = []
    baseline_reach_samples = []
    attempt_results_cm = []
    attempt_shadows = []
    attempt_active = False
    capture_baseline_mode = False
    peak_reach_y = None
    peak_jump_px = 0.0
    jump_in_air = False
    status_message = ""
    status_timer = 0
    calibration_input_mode = False
    calibration_input = ""
    calibration_span_px = None
    shadow_colors = [
        (80, 255, 120),
        (255, 220, 80),
        (255, 140, 80),
        (180, 120, 255),
        (80, 220, 255),
    ]

    window_name = "Vertical Jump Test"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    if os.path.exists(CALIBRATION_FILE):
        print(f"Loaded calibration: {pixel_to_cm:.4f} cm per pixel")
    else:
        print("No saved calibration found. Capture standing reach and press C to calibrate.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame received from the camera.")
                break

            frame = cv.flip(frame, 1)
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = model.detect(mp_image)

            status_lines = [
                f"Attempts: {len(attempt_results_cm)}/{ATTEMPTS_REQUIRED}",
                "S = capture standing reach",
                "C = enter standing reach cm",
                "SPACE = start attempt",
                "R = reset calibration",
                "ESC = exit",
            ]

            if status_timer > 0:
                status_timer -= 1
                status_lines.insert(1, status_message)

            current_reach_y = None
            current_reach_tip = None
            current_left_tip = None
            current_right_tip = None
            current_jump_cm = 0.0
            current_absolute_cm = None

            if results.pose_landmarks:
                landmarks = results.pose_landmarks[0]
                hip_center, ankle_center, nose, head_y = get_body_points(landmarks, width, height)
                reach_tip, left_tip, right_tip = get_reach_points(landmarks, width, height)

                draw_pose(frame, landmarks, width, height)

                cv.circle(frame, tuple(hip_center.astype(int)), 7, (255, 0, 0), -1)
                cv.circle(frame, tuple(ankle_center.astype(int)), 7, (0, 255, 255), -1)
                cv.circle(frame, tuple(nose.astype(int)), 7, (0, 0, 255), -1)
                cv.line(
                    frame,
                    (int(nose[0]) - 25, int(head_y)),
                    (int(nose[0]) + 25, int(head_y)),
                    (0, 0, 255),
                    2,
                )

                if reach_tip is not None:
                    current_reach_tip = reach_tip
                    current_reach_y = float(reach_tip[1])
                    if left_tip is not None:
                        current_left_tip = left_tip
                    if right_tip is not None:
                        current_right_tip = right_tip

                if current_reach_y is not None and baseline_reach_y is not None:
                    current_jump_cm = max(0.0, (baseline_reach_y - current_reach_y) * pixel_to_cm)
                    if standing_reach_cm is not None:
                        current_absolute_cm = standing_reach_cm + current_jump_cm

                if capture_baseline_mode and current_reach_y is not None:
                    baseline_feet_samples.append(ankle_center[1])
                    baseline_reach_samples.append(current_reach_y)

                    if len(baseline_feet_samples) >= BASELINE_FRAMES:
                        baseline_feet_y = float(np.mean(baseline_feet_samples))
                        baseline_reach_y = float(np.mean(baseline_reach_samples))
                        baseline_feet_samples.clear()
                        baseline_reach_samples.clear()
                        capture_baseline_mode = False
                        status_message = "Standing reach captured"
                        status_timer = 120
                        print(
                            f"Standing reach captured: feet y={baseline_feet_y:.2f}px, reach y={baseline_reach_y:.2f}px"
                        )

                if baseline_feet_y is not None and baseline_reach_y is not None:
                    status_lines.insert(
                        1,
                        f"Standing reach locked: {baseline_reach_y:.1f}px above {baseline_feet_y:.1f}px",
                    )

                    if standing_reach_cm is not None:
                        status_lines.insert(
                            2,
                            f"Standing reach: {standing_reach_cm:.1f} cm | live reach: {current_absolute_cm:.1f} cm" if current_absolute_cm is not None else f"Standing reach: {standing_reach_cm:.1f} cm",
                        )
                        status_lines.insert(
                            3,
                            f"Jump above standing reach: {current_jump_cm:.1f} cm",
                        )
                    else:
                        status_lines.insert(
                            2,
                            "Press C and enter your real standing reach in cm",
                        )

                if current_reach_tip is not None:
                    tip_xy = (int(current_reach_tip[0]), int(current_reach_tip[1]))
                    cv.circle(frame, tip_xy, 8, (0, 220, 255), -1)
                    cv.line(frame, (tip_xy[0] - 16, tip_xy[1]), (tip_xy[0] + 16, tip_xy[1]), (0, 220, 255), 2)

                if attempt_active and baseline_reach_y is not None and standing_reach_cm is not None and current_reach_y is not None:
                    if current_jump_cm > peak_jump_px * pixel_to_cm:
                        peak_jump_px = (baseline_reach_y - current_reach_y)
                        peak_reach_y = current_reach_y

                    if current_jump_cm > JUMP_RESET_TOLERANCE:
                        jump_in_air = True

                    if jump_in_air and current_jump_cm < JUMP_RESET_TOLERANCE:
                        jump_height_cm = max(0.0, peak_jump_px * pixel_to_cm)
                        absolute_reach_cm = standing_reach_cm + jump_height_cm
                        attempt_results_cm.append(jump_height_cm)
                        attempt_index = len(attempt_results_cm)
                        attempt_shadows.append(
                            {
                                "attempt": attempt_index,
                                "jump_cm": jump_height_cm,
                                "reach_cm": absolute_reach_cm,
                                "peak_reach_y": peak_reach_y if peak_reach_y is not None else current_reach_y,
                                "color": shadow_colors[(attempt_index - 1) % len(shadow_colors)],
                            }
                        )
                        print(
                            f"Attempt {attempt_index}: standing reach={standing_reach_cm:.2f} cm, jump={jump_height_cm:.2f} cm, total reach={absolute_reach_cm:.2f} cm"
                        )

                        peak_jump_px = 0.0
                        peak_reach_y = None
                        jump_in_air = False
                        attempt_active = False

                        if len(attempt_results_cm) == ATTEMPTS_REQUIRED:
                            average_jump = float(np.mean(attempt_results_cm))
                            max_jump = float(np.max(attempt_results_cm))
                            print("\nVertical Jump Test Complete")
                            print(
                                "Attempts (jump cm): "
                                + ", ".join(f"{value:.2f}" for value in attempt_results_cm)
                            )
                            print(f"Average jump: {average_jump:.2f} cm")
                            print(f"Maximum jump: {max_jump:.2f} cm")
                            stage_result(session_id, attempt_results_cm, standing_reach_cm)
                            break

                    status_lines.insert(
                        1,
                        f"Attempt in progress: peak jump {peak_jump_px * pixel_to_cm:.1f} cm",
                    )

                elif attempt_active and standing_reach_cm is None:
                    status_lines.insert(1, "Calibrate standing reach first")

            status_lines.insert(1, f"Calibration: {pixel_to_cm:.4f} cm/px")

            draw_height_shadows(
                frame,
                baseline_feet_y,
                baseline_reach_y,
                attempt_shadows,
                pixel_to_cm,
            )
            draw_status(frame, status_lines)
            if calibration_input_mode and calibration_span_px is not None:
                draw_calibration_overlay(frame, calibration_input, calibration_span_px)
            cv.imshow(window_name, frame)

            key = cv.waitKey(1) & 0xFF

            if calibration_input_mode:
                if key == 13:
                    try:
                        real_reach_cm = float(calibration_input)
                        span_px = calibration_span_px
                        if span_px and 50 <= real_reach_cm <= 300:
                            standing_reach_cm = real_reach_cm
                            pixel_to_cm = standing_reach_cm / span_px
                            save_calibration(pixel_to_cm)
                            status_message = f"Calibrated: {pixel_to_cm:.4f} cm/px"
                            status_timer = 150
                            print(
                                f"Calibration saved: standing reach={standing_reach_cm:.2f} cm, span={span_px:.2f}px, factor={pixel_to_cm:.4f}"
                            )
                        else:
                            status_message = "Invalid standing reach value"
                            status_timer = 90
                    except ValueError:
                        status_message = "Invalid input, numbers only"
                        status_timer = 90

                    calibration_input_mode = False
                    calibration_input = ""
                    calibration_span_px = None
                elif key == 27:
                    calibration_input_mode = False
                    calibration_input = ""
                    calibration_span_px = None
                    status_message = "Calibration cancelled"
                    status_timer = 90
                elif 48 <= key <= 57:
                    calibration_input += chr(key)
                elif key == 46 and "." not in calibration_input:
                    calibration_input += "."
                elif key == 8:
                    calibration_input = calibration_input[:-1]
                continue

            if key == 27:
                break

            if key in (ord("s"), ord("S")):
                if results.pose_landmarks:
                    baseline_feet_samples.clear()
                    baseline_reach_samples.clear()
                    capture_baseline_mode = True
                    status_message = "Collecting standing reach..."
                    status_timer = 120
                    print("Capturing standing reach baseline. Hold the reach pose steady.")
                else:
                    status_message = "No pose detected for baseline"
                    status_timer = 90

            if (
                key == 32
                and baseline_feet_y is not None
                and baseline_reach_y is not None
                and standing_reach_cm is not None
                and not attempt_active
                and len(attempt_results_cm) < ATTEMPTS_REQUIRED
            ):
                attempt_active = True
                peak_jump_px = 0.0
                peak_reach_y = None
                jump_in_air = False
                print(f"Starting attempt {len(attempt_results_cm) + 1}...")

            if key in (ord("c"), ord("C")):
                if baseline_feet_y is None or baseline_reach_y is None:
                    status_message = "Capture standing reach first"
                    status_timer = 90
                elif attempt_active:
                    status_message = "Finish the current attempt first"
                    status_timer = 90
                else:
                    calibration_input_mode = True
                    calibration_input = ""
                    calibration_span_px = max(1.0, baseline_feet_y - baseline_reach_y)
                    status_message = "Type standing reach height in cm"
                    status_timer = 120
                    print(
                        f"Calibration mode: standing reach span is {calibration_span_px:.2f}px. Enter the real standing reach height in cm."
                    )

            if key in (ord("r"), ord("R")):
                pixel_to_cm = DEFAULT_PIXEL_TO_CM
                standing_reach_cm = None
                baseline_feet_y = None
                baseline_reach_y = None
                baseline_feet_samples.clear()
                baseline_reach_samples.clear()
                attempt_results_cm.clear()
                attempt_shadows.clear()
                attempt_active = False
                capture_baseline_mode = False
                peak_reach_y = None
                peak_jump_px = 0.0
                jump_in_air = False
                calibration_input_mode = False
                calibration_input = ""
                calibration_span_px = None
                if os.path.exists(CALIBRATION_FILE):
                    os.remove(CALIBRATION_FILE)
                status_message = "Calibration reset"
                status_timer = 90
                print("Calibration reset.")

    finally:
        cap.release()
        cv.destroyAllWindows()
        model.close()
        if len(attempt_results_cm) < ATTEMPTS_REQUIRED:
            stage_result(session_id, attempt_results_cm, standing_reach_cm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", type=int, default=None)
    args = parser.parse_args()
    main(session_id=args.session_id)