"""Place the extracted person in a vector eruption.

A shaded cone, a billowing ash column, and a widening lava sheet.
Lava leaves the vent, runs down the face, and spreads on the plain.
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
APEX = (500.0, 148.0)
BASE_Y = 468.0
CRATER = (498, 184)
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


def smoothstep(value):
    shaped = np.clip(value, 0, 1)
    return shaped * shaped * (3 - 2 * shaped)


def build_backdrop(grain):
    """Sky, hazy ridges, and a round cone with gullies and a scree apron."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    lift = yy / (HEIGHT - 1)
    horizon = np.clip((lift - 0.42) / 0.58, 0, 1)
    top = np.array([72, 40, 22], np.float32)
    mid = np.array([108, 72, 46], np.float32)
    haze = np.array([120, 132, 158], np.float32)
    sky = top * (1 - lift[..., None]) + mid * lift[..., None]
    sky = sky * (1 - horizon[..., None] * 0.72) + haze * (horizon[..., None] * 0.72)
    frame = sky.copy()

    def lay_ridge(base, amp, color, cover):
        peaks = (
            base
            + amp * np.sin(np.arange(WIDTH, dtype=np.float32) * 0.012 + base * 0.01)
            + amp * 0.4 * np.sin(np.arange(WIDTH, dtype=np.float32) * 0.029 + 1.4)
        )
        mask = yy > peaks[None, :]
        tone = color * (0.9 + 0.16 * grain)[..., None]
        frame[mask] = frame[mask] * (1 - cover) + tone[mask] * cover

    lay_ridge(292, 34, np.array([96, 90, 84], np.float32), 0.55)
    lay_ridge(338, 22, np.array([62, 60, 58], np.float32), 0.82)

    apex_x, apex_y = APEX
    span = BASE_Y - apex_y
    t = (yy - apex_y) / span
    tt = np.clip(t, 0, 1)
    # Blunt summit so the crater sits in the cone instead of floating on a point.
    half = 48 + 220 * np.power(tt, 1.28)
    skirt = np.power(np.clip((tt - 0.74) / 0.26, 0, 1), 2)
    half = half + skirt * 70
    jag = 1 + 0.04 * np.sin(yy * 0.085 + 0.6) + 0.025 * np.sin(yy * 0.21)
    cx = apex_x - tt * 18 + np.sin(tt * 3.1) * 6
    flank = (xx - cx) / np.maximum(half * jag, 1)
    cone = (t >= 0) & (t <= 1.04) & (np.abs(flank) <= 1) & (yy < HEIGHT - 8)

    facing = np.sqrt(np.clip(1 - np.clip(flank, -1, 1) ** 2, 0, 1))
    light = np.clip(flank * 0.42 + facing * 0.9, 0, 1)
    narrow = cv2.resize(grain, (WIDTH // 7, HEIGHT), interpolation=cv2.INTER_AREA)
    gullies = cv2.resize(narrow, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    gullies = cv2.GaussianBlur(gullies, (9, 9), 0)
    shadow = np.array([26, 24, 30], np.float32)
    lit = np.array([64, 74, 104], np.float32)
    rock = shadow * (1 - light[..., None]) + lit * light[..., None]
    rock *= (0.78 + 0.36 * gullies)[..., None]
    limb = np.clip((np.abs(flank) - 0.78) / 0.22, 0, 1)
    rock *= (1 - limb * 0.5)[..., None]
    trough = np.exp(-((xx - (apex_x - tt * 36)) ** 2) / (2 * (22 + tt * 34) ** 2))
    rock *= (1 - trough * 0.28)[..., None]
    frame[cone] = np.clip(rock, 0, 255)[cone]

    ground = (yy >= 492) & ~cone
    plain = np.array([32, 30, 34], np.float32)
    shade = 1.0 - 0.18 * np.clip((yy - 492) / 48, 0, 1)
    ground_tone = plain * (0.9 + 0.18 * grain)[..., None] * shade[..., None]
    frame[ground] = np.clip(ground_tone, 0, 255)[ground]
    return np.clip(frame, 0, 255).astype(np.uint8)


def add_crater(frame, index):
    """A lava lake in the summit. The near lip is bright and the far lip is dark."""
    pulse = 0.85 + 0.15 * np.sin(index * 0.65)
    rim = (CRATER[0], CRATER[1] - 2)
    cv2.ellipse(frame, rim, (40, 14), 0, 0, 360, (48, 52, 58), -1)
    cv2.ellipse(frame, CRATER, (30, 10), 0, 0, 360, (8, 6, 10), -1)
    lake = (CRATER[0], CRATER[1] + 3)
    orange = (int(20 * pulse), int(80 * pulse), int(220 * pulse))
    cv2.ellipse(frame, lake, (24, 8), 0, 0, 360, orange, -1)
    cv2.ellipse(frame, (CRATER[0], CRATER[1] - 1), (24, 7), 0, 0, 360, (8, 6, 10), -1)
    glow = np.zeros_like(frame, np.float32)
    cv2.ellipse(glow, lake, (28, 10), 0, 0, 360, (16, 70, 200), -1)
    glow = cv2.GaussianBlur(glow, (21, 21), 5)
    frame[:] = np.clip(frame.astype(np.float32) + glow * 0.35, 0, 255).astype(np.uint8)


def add_plume(frame, index, billow):
    """One ash column. Noise bulges the sides so it is not a stack of rings."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    cx, cy = float(CRATER[0]), float(CRATER[1])
    height_frac = np.clip((cy - yy) / 320.0, 0, 1)
    lean = height_frac ** 2 * 96 + np.sin(height_frac * 6.0 + index * 0.22) * (8 + height_frac * 24)
    column_x = cx + lean
    width = 22 + height_frac * 78 + (billow - 0.5) * 36
    dx = (xx - column_x) / np.maximum(width, 8)
    column = np.exp(-0.5 * dx * dx) * np.clip(1.05 - height_frac, 0, 1)
    column *= 0.7 + 0.85 * billow
    column[yy > cy + 8] = 0
    soft = cv2.GaussianBlur(column.astype(np.float32), (21, 21), 5)
    gray = 22 + soft * 110
    ash = np.dstack((gray * 0.75, gray * 0.78, gray * 0.8))
    alpha = np.clip(soft * 1.35, 0, 0.84)[..., None]
    base = frame.astype(np.float32)
    frame[:] = np.clip(base * (1 - alpha) + ash * alpha, 0, 255).astype(np.uint8)


def add_bombs(frame, index):
    """Hot fragments on short arcs, each with a short tail."""
    for step in range(7):
        age = (index * 0.38 + step * 1.37) % 4.6
        direction = (step - 3) * 6.2 + np.sin(step * 1.7) * 3
        speed = 15 + (step % 3) * 3
        x = CRATER[0] + direction * age
        y = CRATER[1] - speed * age + 2.35 * age * age
        earlier = max(0.0, age - 0.28)
        x2 = CRATER[0] + direction * earlier
        y2 = CRATER[1] - speed * earlier + 2.35 * earlier * earlier
        if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
            continue
        cv2.line(frame, (int(x2), int(y2)), (int(x), int(y)), (28, 80, 190), 2, cv2.LINE_AA)
        blend_disc(frame, x, y, 3 + step % 2, (150, 220, 255), 0.95)


def lava_paths(width, height):
    """A front sheet, a left tongue, and a short run down the right face."""
    front = scale_points(
        width,
        height,
        ((0.52, 0.36), (0.48, 0.52), (0.42, 0.68), (0.34, 0.84), (0.28, 0.97)),
    )
    tongue = scale_points(
        width,
        height,
        ((0.49, 0.48), (0.38, 0.64), (0.28, 0.80), (0.20, 0.96)),
    )
    right = scale_points(
        width,
        height,
        ((0.54, 0.38), (0.60, 0.50), (0.66, 0.60)),
    )
    return front, tongue, right


def sheet_masks(dist, along, travel, lobe, crust, scale):
    """An orange sheet. Dark plates float on it and the toe only brightens a little."""
    progress = np.clip(along / max(travel, 1.0), 0, 1)
    radius = (22 + np.power(progress, 1.05) * 96 * scale) * (0.78 + 0.36 * lobe)
    cover = smoothstep((radius - dist) / 18.0)
    center = np.clip(1 - dist / (radius * 0.85 + 1e-3), 0, 1)
    molten = cover * (0.42 + 0.5 * center)
    plates = np.clip((crust - 0.5) * 2.6, 0, 1) * cover
    melt = molten * (1 - plates * 0.8)
    rim = smoothstep((progress - 0.84) / 0.16) * np.clip(1 - dist / (radius * 0.55 + 1e-3), 0, 1)
    melt = np.clip(melt + rim * 0.22, 0, 0.88)
    return melt.astype(np.float32), cover.astype(np.float32)


def paint_lava(frame, cover, melt):
    """Deep red under orange. The glow warms the rock and stays off the white core."""
    picture = frame.astype(np.float32)
    crust = np.array([8, 12, 48], np.float32)
    orange = np.array([18, 72, 220], np.float32)
    yellow = np.array([40, 150, 255], np.float32)
    hot = melt[..., None]
    color = crust * (1 - hot) + orange * hot
    core = np.clip((melt - 0.62) / 0.38, 0, 1)[..., None]
    color = color * (1 - core) + yellow * core
    alpha = cover[..., None]
    mixed = picture * (1 - alpha) + color * alpha
    glow_bgr = (orange / 255.0) * np.clip(melt, 0, 1)[..., None]
    glow_t = torch.from_numpy(np.ascontiguousarray(glow_bgr)).permute(2, 0, 1).unsqueeze(0)
    glow = kornia.filters.gaussian_blur2d(glow_t, (21, 21), (6.0, 6.0))
    base = torch.from_numpy(np.ascontiguousarray(mixed / 255.0)).permute(2, 0, 1).unsqueeze(0)
    lit = 1 - (1 - base) * (1 - glow.clamp(0, 1) * 0.38)
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
    """Warm the side toward the lava. The far side stays cooler."""
    height, width = sprite.shape[:2]
    rgb = sprite[:, :, :3].astype(np.float32)
    across = np.linspace(1.0, 0.2, width, dtype=np.float32)
    down = np.linspace(0.45, 1.0, height, dtype=np.float32)
    light = np.clip(across[None, :] * 0.7 + down[:, None] * 0.3, 0, 1)
    if heat is not None and heat.shape[:2] == (height, width):
        light = np.clip(light * 0.82 + heat * 0.55, 0, 1)
    cool = np.array([0.74, 0.72, 0.68], np.float32)
    warm = np.array([0.84, 0.92, 1.06], np.float32)
    gain = cool + (warm - cool) * light[..., None]
    shaded = np.clip(rgb * gain, 0, 255)
    if heat is not None and heat.shape[:2] == (height, width):
        shaded = np.clip(shaded + heat[..., None] * np.array([4, 16, 36], np.float32), 0, 255)
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


def downhill_noise(field, index, speed, cells):
    """Scroll a field downward and stretch it into long channels."""
    rolled = scrolled(field, index, speed, HEIGHT)
    slim = cv2.resize(rolled, (max(8, WIDTH // cells), max(8, HEIGHT // cells)), interpolation=cv2.INTER_AREA)
    return cv2.resize(slim, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)


def compose_frame(index, sprite, front, tongue, right, lobe_field, crack_field, backdrop):
    frame = backdrop.copy()
    billow = scrolled(lobe_field, index, 1.4, HEIGHT)
    add_plume(frame, index, billow)
    add_crater(frame, index)
    add_bombs(frame, index)
    lobe = downhill_noise(lobe_field, index, 1.0, 10)
    crust = downhill_noise(crack_field, index, 2.6, 5)
    sheets = (
        sheet_masks(front[0], front[1], front[2], lobe, crust, 1.2),
        sheet_masks(tongue[0], tongue[1], tongue[2], lobe, crust, 0.85),
        sheet_masks(right[0], right[1], right[2], lobe, crust, 0.45),
    )
    melt = sheets[0][0]
    cover = sheets[0][1]
    for part_melt, part_cover in sheets[1:]:
        melt = np.maximum(melt, part_melt)
        cover = np.maximum(cover, part_cover)
    frame = paint_lava(frame, cover, melt)
    placed = False
    if sprite is not None:
        sx, sy, sw, sh = sprite_box(WIDTH, HEIGHT, sprite)
        add_contact_shadow(frame, sx, sy, sw, sh)
        near = melt[sy : sy + sh, sx : sx + sw]
        if near.shape[:2] != (sh, sw):
            padded = np.zeros((sh, sw), np.float32)
            padded[: near.shape[0], : near.shape[1]] = near
            near = padded
        placed = paste_person(frame, shade_sprite(sprite, near))
    return frame, placed


def main():
    ready.mkdir(parents=True, exist_ok=True)
    sprite = person_sprite(WIDTH, HEIGHT)
    front_pts, tongue_pts, right_pts = lava_paths(WIDTH, HEIGHT)
    front = flow_field(front_pts, HEIGHT, WIDTH)
    tongue = flow_field(tongue_pts, HEIGHT, WIDTH)
    right = flow_field(right_pts, HEIGHT, WIDTH)
    rng = np.random.default_rng(3)
    grain = build_fbm(HEIGHT, WIDTH, rng, 4)
    backdrop = build_backdrop(grain)
    lobe_field = build_fbm(HEIGHT + 360, WIDTH, rng, 3)
    crack_field = build_fbm(HEIGHT + 360, WIDTH, rng, 5)
    writer, path = open_writer(WIDTH, HEIGHT, FPS)
    preview = None
    placed = False
    try:
        for index in range(FRAMES):
            frame, placed = compose_frame(
                index, sprite, front, tongue, right, lobe_field, crack_field, backdrop
            )
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
