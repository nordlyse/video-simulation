"""Place the extracted person in a vector eruption.

Ash rises from the crater. Lava leaves the vent, runs down the cone,
and spreads on the ground.
"""

from pathlib import Path

import cv2
import kornia
import numpy as np
import torch

ready = Path("/frames/ready")
person_path = ready / "live.png"
video_path = ready / "volcano.mp4"
preview_path = ready / "volcano.png"

WIDTH = 960
HEIGHT = 540
FPS = 12
FRAMES = FPS * 6
CRATER = (480, 198)
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


def blend_disc(frame, x, y, radius, color, alpha):
    if radius < 1 or alpha <= 0:
        return
    height, width = frame.shape[:2]
    x0 = max(0, int(x) - radius)
    y0 = max(0, int(y) - radius)
    x1 = min(width, int(x) + radius + 1)
    y1 = min(height, int(y) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = ((xx - x) ** 2 + (yy - y) ** 2) <= radius * radius
    roi = frame[y0:y1, x0:x1]
    mixed = roi.copy()
    mixed[mask] = (alpha * np.array(color, np.float32) + (1 - alpha) * roi[mask]).astype(np.uint8)
    frame[y0:y1, x0:x1] = mixed


def draw_vector_scene():
    """Sky, ridges, and a cone built from flat shapes."""
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    lift = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None]
    frame[:, :, 0] = (92 * (1 - lift) + 168 * lift).astype(np.uint8)
    frame[:, :, 1] = (62 * (1 - lift) + 118 * lift).astype(np.uint8)
    frame[:, :, 2] = (48 * (1 - lift) + 92 * lift).astype(np.uint8)
    ridge = np.array(
        [[0, 330], [180, 280], [360, 310], [560, 250], [780, 300], [WIDTH, 270], [WIDTH, 430], [0, 430]],
        np.int32,
    )
    cv2.fillPoly(frame, [ridge], (58, 62, 74))
    cone = np.array(
        [[70, 438], [330, 248], [448, 188], [518, 188], [680, 260], [910, 438]],
        np.int32,
    )
    cv2.fillPoly(frame, [cone], (42, 46, 58))
    lit_face = np.array([[480, 190], [518, 188], [760, 438], [470, 438]], np.int32)
    cv2.fillPoly(frame, [lit_face], (36, 48, 72))
    cv2.rectangle(frame, (0, 430), (WIDTH, HEIGHT), (28, 46, 34), -1)
    cv2.ellipse(frame, CRATER, (46, 16), 0, 0, 360, (18, 42, 90), -1)
    cv2.ellipse(frame, CRATER, (28, 9), 0, 0, 360, (30, 90, 255), -1)
    return frame


def add_plume(frame, index):
    """Rising ash. Overlapping puffs are blurred into one column."""
    layer = np.zeros_like(frame)
    cx, cy = CRATER
    life = 8.0
    for step in range(26):
        age = (index * 0.42 + step * (life / 26)) % life
        rise = age / life
        y = int(cy - 6 - rise * (HEIGHT * 0.48))
        spread = 6 + rise * 54
        x = int(cx + np.sin(step * 2.05 + index * 0.18) * spread * 0.28)
        radius = int(22 + rise * 48)
        tone = int(28 + rise * 100)
        color = (max(0, tone - 14), max(0, tone - 6), min(255, tone + 4))
        if radius > 1:
            cv2.circle(layer, (x, y), radius, color, -1)
    soft = cv2.GaussianBlur(layer, (41, 41), 12)
    alpha = np.clip(soft.max(axis=2).astype(np.float32) / 140, 0, 0.78)[..., None]
    frame[:] = (frame.astype(np.float32) * (1 - alpha) + soft.astype(np.float32) * alpha).astype(np.uint8)


def add_bombs(frame, index):
    """Hot fragments thrown from the vent on short arcs."""
    for step in range(9):
        age = (index * 0.35 + step * 1.15) % 5.2
        direction = (step - 4) * 4.4
        x = CRATER[0] + direction * age
        y = CRATER[1] - 16 * age + 2.1 * age * age
        if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
            continue
        blend_disc(frame, x, y, 4, (35, 130, 255), 0.9)


def lava_paths(width, height, sprite):
    slope = scale_points(
        width,
        height,
        ((0.50, 0.38), (0.47, 0.52), (0.42, 0.68), (0.34, 0.86)),
    )
    branch = scale_points(
        width,
        height,
        ((0.50, 0.62), (0.55, 0.78), (0.62, 0.90)),
    )
    if sprite is None:
        front = scale_points(width, height, ((0.62, 0.86), (0.76, 0.92), (0.90, 0.98)))
    else:
        x, y, sprite_width, sprite_height = sprite_box(width, height, sprite)
        front = (
            (x + sprite_width * 0.05, y + sprite_height * 0.82),
            (x + sprite_width * 0.45, y + sprite_height * 0.91),
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


def compose_frame(index, sprite, slope, branch, front, lobe_field, crack_field):
    frame = draw_vector_scene()
    add_plume(frame, index)
    add_bombs(frame, index)
    lobe = scrolled(lobe_field, index, 1.1, HEIGHT)
    cracks = scrolled(crack_field, index, 2.2, HEIGHT)
    frame, slope_heat = lay_flow(frame, slope, 36, lobe, cracks)
    frame, branch_heat = lay_flow(frame, branch, 48, lobe, cracks)
    if sprite is not None:
        sx, sy, sw, sh = sprite_box(WIDTH, HEIGHT, sprite)
        add_contact_shadow(frame, sx, sy, sw, sh)
    front_heat, front_cover = flow_heat(front[0], 22, lobe, cracks)
    placed = False
    if sprite is not None:
        sx, sy, sw, sh = sprite_box(WIDTH, HEIGHT, sprite)
        near = np.maximum(slope_heat[sy : sy + sh, sx : sx + sw], branch_heat[sy : sy + sh, sx : sx + sw])
        near = np.maximum(near, front_heat[sy : sy + sh, sx : sx + sw])
        placed = paste_person(frame, shade_sprite(sprite, near))
    frame = paint_sheet(frame, front_cover, front_heat)
    return frame, placed


def main():
    ready.mkdir(parents=True, exist_ok=True)
    sprite = person_sprite(WIDTH, HEIGHT)
    slope_pts, branch_pts, front_pts = lava_paths(WIDTH, HEIGHT, sprite)
    slope = flow_field(slope_pts, HEIGHT, WIDTH)
    branch = flow_field(branch_pts, HEIGHT, WIDTH)
    front = flow_field(front_pts, HEIGHT, WIDTH)
    rng = np.random.default_rng(3)
    lobe_field = build_fbm(HEIGHT + 360, WIDTH, rng, 3)
    crack_field = build_fbm(HEIGHT + 360, WIDTH, rng, 5)
    writer, path = open_writer(WIDTH, HEIGHT, FPS)
    preview = None
    placed = False
    try:
        for index in range(FRAMES):
            frame, placed = compose_frame(index, sprite, slope, branch, front, lobe_field, crack_field)
            writer.write(frame)
            if index == FRAMES // 2:
                preview = frame
    finally:
        writer.release()

    if preview is None:
        raise SystemExit("The eruption scene produced no frames.")
    cv2.imwrite(str(preview_path), preview)
    print(f"kornia {kornia.__version__}")
    print("person present" if placed else "person absent")
    print(f"vector {WIDTH}x{HEIGHT} frames {FRAMES} fps {FPS}")
    print(f"ready {path}")
    print(f"ready {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
