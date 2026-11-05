"""Place the extracted person in a vector eruption.

A ridged cone, darkened clouds, and lava that breaks as it runs downhill.
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
    half = 78 + 360 * np.power(tt, 1.12)
    skirt = np.power(np.clip((tt - 0.7) / 0.3, 0, 1), 2)
    half = half + skirt * 120
    jag = 1 + 0.02 * np.sin(yy * 0.05 + 0.4) + 0.012 * np.sin(yy * 0.13)
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
    for base_x, amount in ((80, 0.72), (220, -0.5), (360, 0.48), (620, 0.7), (760, -0.58), (900, 0.62)):
        x0, y0 = apex_x, apex_y + 16
        x1, y1 = float(base_x), BASE_Y + 8
        vx, vy = x1 - x0, y1 - y0
        length = float(np.hypot(vx, vy)) + 1e-6
        along = ((xx - x0) * vx + (yy - y0) * vy) / (length * length)
        dist = np.abs((xx - x0) * vy - (yy - y0) * vx) / length
        span_w = 8 + 32 * np.clip(along, 0, 1)
        fall = np.clip(1 - dist / (span_w + 1e-3), 0, 1) ** 2
        band = (along > 0.05) & (along < 1.02) & cone
        shade = np.where(band, fall, 0).astype(np.float32)
        if amount > 0:
            rock *= (1 - shade * amount)[..., None]
        else:
            rock += (shade * (-amount) * 95)[..., None]
    for base_x, color in (
        (120, (14, 12, 16)),
        (260, (76, 86, 122)),
        (400, (12, 10, 14)),
        (760, (82, 92, 128)),
        (880, (14, 12, 16)),
        (980, (78, 88, 120)),
    ):
        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        poly = np.array(
            [
                (int(apex_x) - 4, int(apex_y) + 24),
                (int(apex_x) + 4, int(apex_y) + 24),
                (base_x + 16, int(BASE_Y) + 4),
                (base_x - 12, int(BASE_Y) + 4),
            ],
            np.int32,
        )
        cv2.fillPoly(mask, [poly], 1)
        on_face = (mask > 0) & cone & (np.abs(flank) < 0.78)
        rock[on_face] = np.array(color, np.float32)
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


def add_clouds(frame, index):
    """Soft cumulus banks. Bright tops, gray bases, slow drift."""
    color = np.zeros_like(frame, np.float32)
    cover = np.zeros((HEIGHT, WIDTH), np.float32)
    banks = (
        (120, 70, 0.4),
        (430, 48, 1.2),
        (760, 76, 2.1),
        (250, 150, 0.6),
        (640, 158, 1.7),
    )
    for step, (cx, cy, phase) in enumerate(banks):
        drift = index * (0.28 + (step % 3) * 0.05)
        ox = (cx + drift) % (WIDTH + 240) - 80
        for puff in range(7):
            px = int(ox + (puff - 3) * 46 + np.sin(phase + puff) * 6)
            py = int(cy + 12)
            radius = 42 + (puff % 3) * 10
            cv2.circle(color, (px, py), radius, (78, 82, 90), -1)
            cv2.circle(cover, (px, py), radius, 0.72, -1)
        for puff in range(5):
            px = int(ox + (puff - 2) * 40)
            py = int(cy - 28)
            radius = 30 + puff * 4
            cv2.circle(color, (px, py), radius, (148, 154, 164), -1)
            cv2.circle(cover, (px, py), radius, 0.66, -1)
    color = cv2.GaussianBlur(color, (51, 51), 16)
    cover = cv2.GaussianBlur(cover, (51, 51), 16)
    sky = frame[:, :, 0].astype(np.float32) > frame[:, :, 2].astype(np.float32) + 6
    alpha = np.clip(cover, 0, 0.78) * sky
    alpha = alpha[..., None]
    base = frame.astype(np.float32)
    frame[:] = np.clip(base * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)


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


def paint_chunk(frame, x, y, size, color, spin):
    """An angular lava clot."""
    if size < 1 or not (-size <= x < WIDTH + size and -size <= y < HEIGHT + size):
        return
    pts = []
    for step, ang in enumerate((0.2, 2.3, 4.2)):
        radius = size * (1.05 if step == 0 else 0.72)
        pts.append((int(x + np.cos(ang + spin) * radius), int(y + np.sin(ang + spin) * radius)))
    cv2.fillPoly(frame, [np.array(pts, np.int32)], color)


def add_fountain(frame, index):
    """A lava mushroom that keeps rising. Pieces spill off the cap."""
    cx, cy = float(CRATER[0]), float(CRATER[1])
    cap_y = cy - 132
    stem = np.array(
        [[cx - 18, cy - 4], [cx + 18, cy - 4], [cx + 26, cap_y + 16], [cx - 26, cap_y + 16]],
        np.int32,
    )
    cv2.fillPoly(frame, [stem], (28, 110, 240))
    cv2.ellipse(frame, (int(cx), int(cap_y)), (86, 30), 0, 0, 360, (46, 160, 255), -1)
    cv2.ellipse(frame, (int(cx), int(cap_y - 4)), (48, 16), 0, 0, 360, (90, 210, 255), -1)
    for step in range(16):
        travel = (step * 9 + index * 2.4) % 128
        y = cy - 6 - travel
        if y < cap_y + 8:
            continue
        x = cx + np.sin(step * 1.4 + index * 0.2) * 8
        paint_chunk(frame, x, y, 7, (70, 190, 255), step)
    for step in range(18):
        side = -1.0 if step < 9 else 1.0
        slot = step % 9
        spread = 12 + (slot * 8 + index * 1.6) % 74
        x = cx + side * spread
        y = cap_y - 6 + (spread / 86) * 10
        paint_chunk(frame, x, y, 8, (64, 184, 255), step * 0.5)
    for side in (-1.0, 1.0):
        for slot in range(7):
            fall = (slot * 16 + index * 1.15) % 120
            x = cx + side * (78 + fall * 0.22)
            y = cap_y + 8 + 0.011 * fall * fall + slot * 3
            if y > cy + 150:
                continue
            paint_chunk(frame, x, y, 8, (32, 120, 230), slot + side)


def lava_channels(width, height):
    """Channels leave the crater lip and only step downhill."""
    return (
        (
            scale_points(width, height, ((0.52, 0.31), (0.49, 0.46), (0.45, 0.62), (0.39, 0.80), (0.33, 0.98))),
            16,
            24,
            0.4,
        ),
        (
            scale_points(width, height, ((0.53, 0.32), (0.50, 0.50), (0.46, 0.68), (0.41, 0.84), (0.37, 0.99))),
            12,
            18,
            1.2,
        ),
        (
            scale_points(width, height, ((0.50, 0.36), (0.42, 0.54), (0.32, 0.74), (0.24, 0.98))),
            11,
            16,
            2.0,
        ),
        (
            scale_points(width, height, ((0.47, 0.58), (0.38, 0.76), (0.31, 0.98))),
            9,
            13,
            2.7,
        ),
        (
            scale_points(width, height, ((0.54, 0.32), (0.57, 0.42), (0.60, 0.50), (0.63, 0.58))),
            10,
            14,
            0.6,
        ),
    )


def path_samples(points, count):
    pts = np.asarray(points, np.float32)
    if len(pts) < 2:
        return pts, np.zeros(len(pts), np.float32), np.zeros((len(pts), 2), np.float32)
    steps = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    travel = np.concatenate([[0], np.cumsum(steps)])
    total = float(travel[-1])
    marks = np.linspace(0, total, count)
    pos = np.empty((count, 2), np.float32)
    tangent = np.empty((count, 2), np.float32)
    progress = marks / max(total, 1e-6)
    for i, mark in enumerate(marks):
        k = int(np.searchsorted(travel, mark, side="right") - 1)
        k = min(max(k, 0), len(pts) - 2)
        span = float(travel[k + 1] - travel[k])
        blend = 0.0 if span < 1e-4 else (mark - float(travel[k])) / span
        pos[i] = pts[k] * (1 - blend) + pts[k + 1] * blend
        direction = pts[k + 1] - pts[k]
        tangent[i] = direction / (float(np.hypot(direction[0], direction[1])) + 1e-6)
    return pos, progress, tangent


def meander(pos, phase):
    """Bend the channel sideways. Y only increases, so the flow runs downhill."""
    out = pos.copy()
    for i in range(len(out)):
        shift = np.sin(i * 0.42 + phase) * 7 + np.sin(i * 1.05 + phase * 1.2) * 3
        out[i, 0] += shift
    for i in range(1, len(out)):
        if out[i, 1] < out[i - 1, 1] + 3:
            out[i, 1] = out[i - 1, 1] + 3
    return out


def tangents_of(pos):
    tangent = np.zeros_like(pos)
    if len(pos) < 2:
        return tangent
    for i in range(len(pos) - 1):
        direction = pos[i + 1] - pos[i]
        tangent[i] = direction / (float(np.hypot(direction[0], direction[1])) + 1e-6)
    tangent[-1] = tangent[-2]
    return tangent


def crust_slabs(pos, progress, tangent, start_w, end_w, index):
    """Irregular crust chunks beside a molten lane that keeps running downhill."""
    slabs = []
    phase = index * 0.08
    i = 1
    while i < len(pos) - 2:
        span = 2 + int((0.5 + 0.5 * np.sin(i * 0.85 + phase)) * 4)
        j = min(i + span, len(pos) - 1)
        side = 1.0 if np.sin(i * 1.37 + phase) > -0.1 else -1.0
        cover = 0.34 + 0.22 * (0.5 + 0.5 * np.sin(i * 0.6 + phase))

        def edge(k):
            width = (start_w + (end_w - start_w) * float(progress[k])) * cover
            nx, ny = -float(tangent[k, 1]), float(tangent[k, 0])
            cx = pos[k, 0] + nx * width * 0.85 * side
            cy = pos[k, 1] + ny * width * 0.85 * side
            return (cx + nx * width, cy + ny * width), (cx - nx * width, cy - ny * width)

        top_a, top_b = edge(i)
        bot_a, bot_b = edge(j)
        slabs.append(np.array([top_a, bot_a, bot_b, top_b], np.int32))
        gap = 1 + int((0.5 + 0.5 * np.sin(i * 2.2 + phase)) * 2)
        i = j + gap
    return slabs


def jagged_strip(pos, progress, tangent, start_w, end_w, phase, inner):
    """Even width, a small torn edge, and a pointed end."""
    left = []
    right = []
    for i in range(len(pos)):
        nx = -float(tangent[i, 1])
        ny = float(tangent[i, 0])
        width = start_w + (end_w - start_w) * float(progress[i])
        if progress[i] > 0.88:
            width *= 1 - (float(progress[i]) - 0.88) / 0.12 * 0.75
        if inner:
            width *= 0.46
        jag = np.sin(i * 2.4 + phase) * 0.1 + np.sin(i * 5.2 + phase * 1.3) * 0.05
        half = max(1.8, width * (1.0 + jag))
        left.append((pos[i, 0] + nx * half, pos[i, 1] + ny * half))
        right.append((pos[i, 0] - nx * half, pos[i, 1] - ny * half))
    if len(left) < 2:
        return None
    return np.array(left + right[::-1], np.int32)


def paint_channels(frame, index):
    """Downhill channels. Crust plates ride on the molten core."""
    crust_color = (14, 16, 42)
    molten_color = (22, 96, 235)
    toe_color = (36, 150, 255)
    slab_color = (10, 12, 28)
    heat = np.zeros(frame.shape[:2], np.float32)
    banks = []
    cores = []
    toes = []
    slabs = []
    for points, start_w, end_w, phase in lava_channels(WIDTH, HEIGHT):
        pos, progress, _tangent = path_samples(points, 48)
        pos = meander(pos, phase + index * 0.025)
        tangent = tangents_of(pos)
        bank = jagged_strip(pos, progress, tangent, start_w, end_w, phase, False)
        core = jagged_strip(pos, progress, tangent, start_w, end_w, phase, True)
        if bank is not None:
            banks.append(bank)
        if core is not None:
            cores.append(core)
        cut = len(pos) // 2
        toe = jagged_strip(pos[cut:], progress[cut:], tangent[cut:], start_w, end_w, phase, True)
        if toe is not None:
            toes.append(toe)
        slabs.extend(crust_slabs(pos, progress, tangent, start_w, end_w, index))
    if banks:
        cv2.fillPoly(frame, banks, crust_color)
    if cores:
        cv2.fillPoly(frame, cores, molten_color)
        cv2.fillPoly(heat, cores, 1)
    if toes:
        cv2.fillPoly(frame, toes, toe_color)
    if slabs:
        cv2.fillPoly(frame, slabs, slab_color)
        cv2.fillPoly(heat, slabs, 0.12)
    fringe = cv2.GaussianBlur(heat, (5, 5), 1.1)
    fringe = np.clip(fringe - heat, 0, 1)
    warm = fringe[..., None] * np.array([10, 40, 90], np.float32)
    frame[:] = np.clip(frame.astype(np.float32) + warm, 0, 255).astype(np.uint8)
    return heat


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


def compose_frame(index, sprite, lobe_field, backdrop):
    frame = backdrop.copy()
    add_clouds(frame, index)
    billow = scrolled(lobe_field, index, 1.4, HEIGHT)
    add_plume(frame, index, billow)
    add_crater(frame, index)
    melt = paint_channels(frame, index)
    add_fountain(frame, index)
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
    rng = np.random.default_rng(3)
    grain = build_fbm(HEIGHT, WIDTH, rng, 4)
    backdrop = build_backdrop(grain)
    lobe_field = build_fbm(HEIGHT + 360, WIDTH, rng, 3)
    writer, path = open_writer(WIDTH, HEIGHT, FPS)
    preview = None
    placed = False
    try:
        for index in range(FRAMES):
            frame, placed = compose_frame(index, sprite, lobe_field, backdrop)
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
