"""Place the extracted person in a short recording of a real eruption.

The background is the 8 April 2025 Kanlaon eruption, filmed by the
PHIVOLCS camera at Lower Masulog, Canlaon City. The recording is a
Philippine government work and is in the public domain.

https://commons.wikimedia.org/wiki/File:An_explosive_eruption_occurred_at_Kanlaon_Volcano_on_April_8,_2025_(full_view).webm
"""

from pathlib import Path

import cv2
import kornia
import numpy as np

ready = Path("/frames/ready")
person_path = ready / "live.png"
background_path = Path("/media/kanlaon.webm")
video_path = ready / "volcano.mp4"
preview_path = ready / "volcano.png"

# The plume is rising through this part of the recording.
START_SEC = 10
SECONDS = 8
# A real person covers more of the cutout than mask noise.
MIN_PERSON = 0.005


def person_sprite(frame_width, frame_height):
    if not person_path.is_file():
        raise SystemExit(f"Missing person frame: {person_path}")
    image = cv2.imread(str(person_path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise SystemExit(f"Unexpected person frame: {person_path}")
    present = image[:, :, 3] > 20
    if float(present.mean()) < MIN_PERSON:
        return None
    ys, xs = np.where(present)
    sprite = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    target = int(frame_height * 0.46)
    scale = target / sprite.shape[0]
    if sprite.shape[1] * scale > frame_width * 0.42:
        scale = (frame_width * 0.42) / sprite.shape[1]
    size = (max(1, int(sprite.shape[1] * scale)), max(1, int(sprite.shape[0] * scale)))
    return cv2.resize(sprite, size, interpolation=cv2.INTER_AREA)


def paste_person(frame, sprite):
    if sprite is None:
        return False
    frame_height, frame_width = frame.shape[:2]
    height, width = sprite.shape[:2]
    x = frame_width - width - int(frame_width * 0.08)
    y = frame_height - height
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_width, x + width)
    y1 = min(frame_height, y + height)
    cut = sprite[y0 - y : y0 - y + (y1 - y0), x0 - x : x0 - x + (x1 - x0)]
    color = cut[:, :, :3].astype(np.float32)
    alpha = cut[:, :, 3:4].astype(np.float32) / 255
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (color * alpha + roi * (1 - alpha)).astype(np.uint8)
    return True


def open_writer(width, height, fps):
    for codec, suffix in (("mp4v", ".mp4"), ("MJPG", ".avi")):
        path = video_path.with_suffix(suffix)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer, path
        writer.release()
    raise SystemExit("The video writer did not open.")


def main():
    if not background_path.is_file():
        raise SystemExit(f"Missing volcano recording: {background_path}")
    ready.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(background_path))
    if not capture.isOpened():
        raise SystemExit(f"The volcano recording did not open: {background_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    if fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width < 2 or height < 2:
        raise SystemExit("The volcano recording has no picture.")

    sprite = person_sprite(width, height)
    capture.set(cv2.CAP_PROP_POS_MSEC, START_SEC * 1000)
    frame_count = int(round(SECONDS * fps))
    writer, path = open_writer(width, height, fps)
    preview = None
    placed = False
    written = 0
    try:
        for index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                break
            placed = paste_person(frame, sprite)
            writer.write(frame)
            written += 1
            if index == frame_count // 2:
                preview = frame.copy()
    finally:
        capture.release()
        writer.release()

    if written == 0 or preview is None:
        raise SystemExit("The volcano recording produced no frames.")
    cv2.imwrite(str(preview_path), preview)
    print(f"kornia {kornia.__version__}")
    print("person present" if placed else "person absent")
    print(f"kanlaon {width}x{height} frames {written} fps {fps:.2f}")
    print(f"ready {path}")
    print(f"ready {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
