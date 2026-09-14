"""Turning a camera *source* into an open capture, in one place.

Every engine used to decide this for itself: the dual-feed ones parsed a
URL-or-index (they had to, the coach types one into the form), while the six
single-camera engines hardwired VideoCapture(0). That hardcode is the reason
two coaches could never run tests at the same time — not the laptop having
one webcam, but the code being unable to point anywhere else.

A source is whatever the coach configured at sign-in:

    "0", 0        the laptop's built-in camera
    "1"           a second attached camera
    "http://…"    a phone running an IP-camera app, which is what makes two
                  coaches on one server actually possible
"""

import cv2


def resolve_source(source):
    """Normalise to what cv2.VideoCapture wants: int index or str URL.

    Digit strings become ints — "0" from a form field and 0 from a default
    have to mean the same camera, or the guard that stops two coaches
    opening one device would compare them as different.
    """
    if source is None:
        return 0
    if isinstance(source, int):
        return source
    text = str(source).strip()
    if not text:
        return 0
    return int(text) if text.isdigit() else text


def source_key(source):
    """Stable identity for a source, for 'is another coach already using this
    camera?' checks. Two spellings of the same device must collide."""
    resolved = resolve_source(source)
    return str(resolved).rstrip("/").lower() if isinstance(resolved, str) else f"index:{resolved}"


def mirror(frame, enabled=True):
    """Selfie-flip a frame, unless the camera already did it.

    Every engine mirrors its preview so a subject facing the camera sees
    their real right hand on the right. That assumes the camera hands over a
    plain, un-mirrored image — true of a USB webcam, NOT true of some phone
    sources. iPhone Continuity Camera and several IP-camera apps mirror the
    stream themselves, and our flip then cancels theirs: the coach steps
    left and the picture moves right, which is exactly the complaint
    mirroring exists to prevent.

    Nothing can detect this from the pixels, so it is a per-camera setting
    the coach makes once at sign-in.
    """
    return cv2.flip(frame, 1) if enabled else frame


_OTHER_SIDE = {"Left": "Right", "Right": "Left"}


def anatomical_side(label, mirrored=True):
    """Correct a MediaPipe Left/Right label for a mirrored frame.

    MediaPipe labels what it SEES. The engines flip before inference — they
    must, so landmarks, overlays and calibration clicks share the displayed
    coordinate space — but that hands the model a mirror-image of the person.
    It then labels the subject's real RIGHT arm as "Left", because in a
    mirrored picture that is what a left arm looks like.

    The measurements themselves are unaffected: wingspan is a span between
    two tips, its gate wants BOTH arms at 90 degrees, and wall_toss counts
    alternation — all symmetric. What breaks is what the screen TELLS the
    student, which is the opposite of the arm they should move.
    """
    return _OTHER_SIDE.get(label, label) if mirrored else label


def open_capture(source, width=None, height=None, fps=None, buffersize=1):
    """Open a capture, or raise with the source named.

    The old message was a bare "Cannot open camera (index 0)" even when the
    coach had pointed the app at a phone URL, which sent people looking at
    the wrong device entirely.
    """
    resolved = resolve_source(source)

    cap = cv2.VideoCapture(resolved, cv2.CAP_ANY)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(resolved)  # retry without the backend hint
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Cannot open camera {resolved!r} — check it is connected, not in "
            "use by another app or another coach, and that the URL loads in a browser."
        )

    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    if buffersize:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffersize)
    return cap
