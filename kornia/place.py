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
import torch

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


def scale_points(width, height, pairs):
    return [(x * width, y * height) for x, y in pairs]


def flow_field(points, height, width):
    """Distance from each pixel to a lava channel, and distance along it."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dist = np.full((height, width), np.float32(1e6))
    along = np.zeros((height, width), np.float32)
    travel = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        vx = float(x1 - x0)
        vy = float(y1 - y0)
        length = float(np.hypot(vx, vy)) + 1e-6
        t = ((xx - x0) * vx + (yy - y0) * vy) / (length * length)
        np.clip(t, 0, 1, out=t)
        px = x0 + t * vx
        py = y0 + t * vy
        segment = np.hypot(xx - px, yy - py).astype(np.float32)
        closer = segment < dist
        np.copyto(dist, segment, where=closer)
        np.copyto(along, np.float32(travel) + t * length, where=closer)
        travel += length
    return dist, along, travel


def build_fbm(height, width, rng, octaves):
    """Layered noise. Low octaves are broad lobes, high octaves are cracks."""
    total = np.zeros((height, width), np.float32)
    weight = 0.0
    amplitude = 1.0
    for level in range(octaves):
        cells_y = max(2, height // (2 ** (level + 3)))
        cells_x = max(2, width // (2 ** (level + 3)))
        grid = rng.random((cells_y, cells_x), dtype=np.float32)
        layer = cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC)
        total += layer * amplitude
        weight += amplitude
        amplitude *= 0.55
    return total / weight


def scrolled(field, index, speed, height):
    rolled = np.roll(field, int(index * speed), axis=0)
    return rolled[:height]


def flow_heat(dist, radius, lobe, cracks):
    """Dark crust with a few molten channels running downhill."""
    limit = radius * (0.72 + 0.5 * lobe)
    cover = np.clip((limit - dist) / (radius * 0.28), 0, 1)
    cover[dist >= limit] = 0
    height, width = cracks.shape
    slim = cv2.resize(cracks, (max(8, width // 7), height), interpolation=cv2.INTER_AREA)
    rivers = cv2.resize(slim, (width, height), interpolation=cv2.INTER_LINEAR)
    heat = np.clip((rivers - 0.57) * 5.0, 0, 1)
    heat[cover < 0.45] = 0
    return heat.astype(np.float32), cover.astype(np.float32)


def paint_sheet(frame, cover, heat):
    """Darken the ground into crust and light only the cracks."""
    picture = frame.astype(np.float32)
    crust = picture * np.array([0.28, 0.22, 0.42], np.float32)
    glow_color = np.array([40, 120, 255], np.float32)
    core_color = np.array([170, 235, 255], np.float32)
    melt = glow_color * (1 - heat[..., None]) + core_color * heat[..., None]
    lava = crust * (1 - heat[..., None]) + melt * heat[..., None]
    cover_c = np.clip(cover, 0, 1)[..., None]
    mixed = picture * (1 - cover_c) + lava * cover_c
    hot = (melt / 255) * heat[..., None]
    hot_t = torch.from_numpy(np.ascontiguousarray(hot)).permute(2, 0, 1).unsqueeze(0)
    glow = kornia.filters.gaussian_blur2d(hot_t, (21, 21), (6.0, 6.0))
    base = torch.from_numpy(np.ascontiguousarray(mixed / 255)).permute(2, 0, 1).unsqueeze(0)
    lit = 1 - (1 - base) * (1 - glow.clamp(0, 1) * 0.65)
    return (lit.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()


def add_contact_shadow(frame, x, y, width, height):
    """Darken the ground to the right of the person, away from the lava."""
    frame_height, frame_width = frame.shape[:2]
    center = (int(x + width * 0.62), int(min(frame_height - 6, y + height - 4)))
    axes = (max(8, int(width * 0.34)), max(4, int(height * 0.07)))
    mask = np.zeros(frame.shape[:2], np.float32)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 8)
    y0 = max(0, center[1] - axes[1] - 20)
    x0 = max(0, center[0] - axes[0] - 20)
    y1 = min(frame_height, center[1] + axes[1] + 20)
    x1 = min(frame_width, center[0] + axes[0] + 20)
    shade = mask[y0:y1, x0:x1, None]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = np.clip(roi * (1 - shade * 0.55), 0, 255).astype(np.uint8)


def shade_sprite(sprite, heat):
    """Warm the cutout from the lava and darken the side facing away from it."""
    height, width = sprite.shape[:2]
    rgb = sprite[:, :, :3].astype(np.float32)
    across = np.linspace(1.0, 0.0, width, dtype=np.float32)
    down = np.linspace(0.35, 1.0, height, dtype=np.float32)
    light = np.clip(across[None, :] * 0.72 + down[:, None] * 0.48, 0, 1)
    if heat is not None and heat.shape[:2] == (height, width):
        light = np.clip(light * 0.62 + heat * 1.15, 0, 1)
    cool = np.array([0.78, 0.72, 0.62], np.float32)
    warm = np.array([0.88, 0.98, 1.16], np.float32)
    gain = cool + (warm - cool) * light[..., None]
    shaded = np.clip(rgb * gain + light[..., None] * np.array([2, 10, 22], np.float32), 0, 255)
    if heat is not None and heat.shape[:2] == (height, width):
        shaded = np.clip(shaded + heat[..., None] * np.array([8, 36, 78], np.float32), 0, 255)

    mask = np.where(sprite[:, :, 3] > 20, np.uint8(255), np.uint8(0))
    inner = cv2.erode(mask, np.ones((3, 3), np.uint8))
    rim = ((mask > 0) & (inner == 0)).astype(np.float32) * np.clip(across[None, :] - 0.35, 0, 1)
    shaded = np.clip(shaded + rim[..., None] * np.array([12, 48, 110], np.float32), 0, 255)

    out = sprite.copy()
    out[:, :, :3] = shaded.astype(np.uint8)
    return out


def lava_paths(width, height, sprite):
    slope = scale_points(width, height, ((0.48, 0.56), (0.50, 0.62), (0.53, 0.68)))
    branch = scale_points(width, height, ((0.50, 0.58), (0.56, 0.64), (0.61, 0.70)))
    if sprite is None:
        front = scale_points(width, height, ((0.68, 0.88), (0.80, 0.93), (0.92, 0.97)))
    else:
        x, y, sprite_width, sprite_height = sprite_box(width, height, sprite)
        front = (
            (x + sprite_width * 0.08, y + sprite_height * 0.84),
            (x + sprite_width * 0.48, y + sprite_height * 0.92),
            (x + sprite_width * 0.88, y + sprite_height * 0.98),
        )
    return slope, branch, front


def lay_flow(frame, field, radius, lobe, cracks):
    heat, cover = flow_heat(field[0], radius, lobe, cracks)
    return paint_sheet(frame, cover, heat), heat


def sprite_box(frame_width, frame_height, sprite):
    height, width = sprite.shape[:2]
    x = frame_width - width - int(frame_width * 0.08)
    y = frame_height - height
    return x, y, width, height


def paste_person(frame, sprite):
    if sprite is None:
        return False
    frame_height, frame_width = frame.shape[:2]
    x, y, width, height = sprite_box(frame_width, frame_height, sprite)
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
    slope_pts, branch_pts, front_pts = lava_paths(width, height, sprite)
    slope = flow_field(slope_pts, height, width)
    branch = flow_field(branch_pts, height, width)
    front = flow_field(front_pts, height, width)
    rng = np.random.default_rng(3)
    lobe_field = build_fbm(height + 480, width, rng, 3)
    crack_field = build_fbm(height + 480, width, rng, 5)
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
            lobe = scrolled(lobe_field, index, 1.2, height)
            cracks = scrolled(crack_field, index, 2.4, height)
            frame, slope_heat = lay_flow(frame, slope, 78, lobe, cracks)
            frame, branch_heat = lay_flow(frame, branch, 46, lobe, cracks)
            if sprite is not None:
                sx, sy, sw, sh = sprite_box(width, height, sprite)
                add_contact_shadow(frame, sx, sy, sw, sh)
            front_heat, front_cover = flow_heat(front[0], 34, lobe, cracks)
            if sprite is not None:
                sx, sy, sw, sh = sprite_box(width, height, sprite)
                near = np.maximum(slope_heat[sy : sy + sh, sx : sx + sw], branch_heat[sy : sy + sh, sx : sx + sw])
                near = np.maximum(near, front_heat[sy : sy + sh, sx : sx + sw])
                placed = paste_person(frame, shade_sprite(sprite, near))
            frame = paint_sheet(frame, front_cover, front_heat)
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
