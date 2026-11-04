"""Place the extracted person in a short erupting-volcano scene.

The scene is derived from the MIT-licensed animation at
https://github.com/bsagevedant/Volcanic-Simulation

Copyright (c) 2025 bhskr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from pathlib import Path

import cv2
import kornia
import numpy as np

ready = Path("/frames/ready")
person_path = ready / "live.png"
video_path = ready / "volcano.mp4"
preview_path = ready / "volcano.png"

WIDTH = 960
HEIGHT = 540
FPS = 12
FRAMES = FPS * 5
SX = WIDTH / 600
SY = HEIGHT / 400
# A real person covers more of the cutout than mask noise.
MIN_PERSON = 0.005


def px(x, y):
    return int(round(x * SX)), int(round(y * SY))


def unit(value):
    return max(1, int(round(value * (SX + SY) / 2)))


def blend_disc(frame, x, y, radius, color, alpha):
    if radius < 1 or alpha <= 0:
        return
    height, width = frame.shape[:2]
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(width, x + radius + 1)
    y1 = min(height, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius * radius
    roi = frame[y0:y1, x0:x1]
    mixed = roi.copy()
    mixed[mask] = (
        alpha * np.array(color, dtype=np.float32) + (1 - alpha) * roi[mask]
    ).astype(np.uint8)
    frame[y0:y1, x0:x1] = mixed


def scene_base():
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:] = (60, 25, 25)
    for index in range(100):
        x = (index * 37) % WIDTH
        y = (index * 23) % (HEIGHT // 2)
        cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
    mountain = np.array(
        [px(-50, 400), px(200, 200), px(400, 200), px(650, 400)],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [mountain], (40, 50, 60))
    crater = px(300, 200)
    cv2.ellipse(frame, crater, (unit(100), unit(20)), 0, 0, 360, (20, 30, 80), -1)
    cv2.ellipse(frame, crater, (unit(80), unit(15)), 0, 0, 360, (0, 80, 255), -1)
    return frame


def add_particle(rng):
    return {
        "x": 300 + float(rng.uniform(-20, 20)),
        "y": 200.0,
        "vx": float(rng.uniform(-2, 2)),
        "vy": float(rng.uniform(-12, -8)),
        "size": float(rng.uniform(5, 15)),
        "color": (
            int(rng.uniform(0, 50)),
            int(rng.uniform(100, 200)),
            255,
        ),
        "life": 255,
    }


def add_smoke(rng):
    return {
        "x": 300 + float(rng.uniform(-15, 15)),
        "y": 195.0,
        "vx": float(rng.uniform(-0.5, 0.5)),
        "vy": float(rng.uniform(-2, -1)),
        "size": float(rng.uniform(20, 40)),
        "opacity": float(rng.uniform(0.35, 0.55)),
        "expand": float(rng.uniform(0.3, 0.8)),
        "life": 255,
    }


def add_lava_flow(rng):
    return {
        "x": 300 + float(rng.uniform(-50, 50)),
        "y": 200.0,
        "width": float(rng.uniform(10, 25)),
        "speed": float(rng.uniform(0.5, 1.5)),
        "points": [],
        "max_length": int(rng.uniform(50, 150)),
    }


def draw_crater(frame, rng):
    for _ in range(5):
        x, y = px(300 + float(rng.uniform(-70, 70)), 200 + float(rng.uniform(-10, 10)))
        blend_disc(
            frame,
            x,
            y,
            unit(float(rng.uniform(5, 15))),
            (0, 150, 255),
            0.75,
        )


def step_particles(particles, frame, rng):
    alive = []
    for particle in particles:
        particle["vy"] += 0.2
        particle["x"] += particle["vx"] + float(rng.uniform(-0.5, 0.5))
        particle["y"] += particle["vy"]
        particle["life"] -= 2
        point = px(particle["x"], particle["y"])
        radius = unit(particle["size"])
        blend_disc(frame, point[0], point[1], int(radius * 1.5), (0, 150, 255), 0.2)
        cv2.circle(frame, point, radius, particle["color"], -1)
        if particle["life"] > 0 and particle["y"] <= 400:
            alive.append(particle)
    particles[:] = alive


def step_smoke(puffs, frame):
    alive = []
    for puff in puffs:
        puff["x"] += puff["vx"]
        puff["y"] += puff["vy"]
        puff["size"] += puff["expand"]
        puff["life"] -= 1
        point = px(puff["x"], puff["y"])
        alpha = puff["opacity"] * (puff["life"] / 255)
        blend_disc(frame, point[0], point[1], unit(puff["size"]), (100, 100, 100), alpha)
        if puff["life"] > 0:
            alive.append(puff)
    puffs[:] = alive


def step_lava(flows, frame, rng, index):
    alive = []
    for flow in flows:
        if len(flow["points"]) == 0 or (index % 2 == 0 and len(flow["points"]) < flow["max_length"]):
            flow["points"].append(
                {
                    "x": flow["x"] + float(rng.uniform(-5, 5)),
                    "y": flow["y"],
                    "width": flow["width"],
                }
            )
            flow["y"] += flow["speed"]
        points = flow["points"]
        span = max(1, len(points) - 1)
        for rank, (left, right) in enumerate(zip(points, points[1:])):
            alpha = 1 - (rank / span) * 0.6
            start = px(left["x"], left["y"])
            end = px(right["x"], right["y"])
            thickness = max(1, unit(left["width"]))
            blend_line(frame, start, end, int(thickness * 1.5), (0, 150, 255), alpha * 0.4)
            blend_line(frame, start, end, thickness, (0, 100, 255), alpha)
        if len(points) >= flow["max_length"]:
            for point in points:
                point["width"] *= 0.98
            if points[0]["width"] >= 1:
                alive.append(flow)
        else:
            alive.append(flow)
    flows[:] = alive


def blend_line(frame, start, end, thickness, color, alpha):
    if thickness < 1 or alpha <= 0:
        return
    pad = thickness + 2
    x0 = max(0, min(start[0], end[0]) - pad)
    y0 = max(0, min(start[1], end[1]) - pad)
    x1 = min(frame.shape[1], max(start[0], end[0]) + pad + 1)
    y1 = min(frame.shape[0], max(start[1], end[1]) + pad + 1)
    if x0 >= x1 or y0 >= y1:
        return
    roi = frame[y0:y1, x0:x1]
    overlay = roi.copy()
    cv2.line(
        overlay,
        (start[0] - x0, start[1] - y0),
        (end[0] - x0, end[1] - y0),
        color,
        thickness,
        cv2.LINE_AA,
    )
    mask = np.any(overlay != roi, axis=2)
    roi[mask] = (
        alpha * overlay[mask].astype(np.float32) + (1 - alpha) * roi[mask].astype(np.float32)
    ).astype(np.uint8)


def person_sprite():
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
    target = int(HEIGHT * 0.62)
    scale = target / sprite.shape[0]
    if sprite.shape[1] * scale > WIDTH * 0.7:
        scale = (WIDTH * 0.7) / sprite.shape[1]
    size = (max(1, int(sprite.shape[1] * scale)), max(1, int(sprite.shape[0] * scale)))
    return cv2.resize(sprite, size, interpolation=cv2.INTER_AREA)


def paste_person(frame, sprite):
    if sprite is None:
        return False
    height, width = sprite.shape[:2]
    x = (WIDTH - width) // 2
    y = HEIGHT - height - 8
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(WIDTH, x + width)
    y1 = min(HEIGHT, y + height)
    cut = sprite[y0 - y : y0 - y + (y1 - y0), x0 - x : x0 - x + (x1 - x0)]
    color = cut[:, :, :3].astype(np.float32)
    alpha = cut[:, :, 3:4].astype(np.float32) / 255
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (color * alpha + roi * (1 - alpha)).astype(np.uint8)
    return True


def open_writer():
    for codec, suffix in (("mp4v", ".mp4"), ("MJPG", ".avi")):
        path = video_path.with_suffix(suffix)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            FPS,
            (WIDTH, HEIGHT),
        )
        if writer.isOpened():
            return writer, path
        writer.release()
    raise SystemExit("The video writer did not open.")


def main():
    ready.mkdir(parents=True, exist_ok=True)
    sprite = person_sprite()
    rng = np.random.default_rng(7)
    base = scene_base()
    particles = [add_particle(rng) for _ in range(20)]
    puffs = [add_smoke(rng) for _ in range(10)]
    flows = []
    writer, path = open_writer()
    preview = None
    placed = False
    try:
        for index in range(FRAMES):
            frame = base.copy()
            draw_crater(frame, rng)
            step_lava(flows, frame, rng, index)
            step_particles(particles, frame, rng)
            step_smoke(puffs, frame)
            if index % 3 == 0:
                particles.append(add_particle(rng))
            if index % 5 == 0:
                puffs.append(add_smoke(rng))
            if index % 20 == 0 and float(rng.random()) < 0.5:
                flows.append(add_lava_flow(rng))
            placed = paste_person(frame, sprite)
            writer.write(frame)
            if index == FRAMES // 2:
                preview = frame.copy()
    finally:
        writer.release()

    if preview is None:
        raise SystemExit("The volcano scene produced no frames.")
    cv2.imwrite(str(preview_path), preview)
    print(f"kornia {kornia.__version__}")
    print("person present" if placed else "person absent")
    print(f"scene {WIDTH}x{HEIGHT} frames {FRAMES} fps {FPS}")
    print(f"ready {path}")
    print(f"ready {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
