"""Place the extracted person in a vector eruption.

A ridged cone, darkened clouds, and lava that branches downhill.
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
    """Sky, hazy ridges, a round cone, and a meadow in front of it."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    lift = yy / (HEIGHT - 1)
    horizon = np.clip((lift - 0.42) / 0.58, 0, 1)
    top = np.array([72, 40, 22], np.float32)
    mid = np.array([108, 72, 46], np.float32)
    haze = np.array([120, 132, 158], np.float32)
    sky = top * (1 - lift[..., None]) + mid * lift[..., None]
    sky = sky * (1 - horizon[..., None] * 0.72) + haze * (horizon[..., None] * 0.72)
    frame = sky.copy()

    def lay_ridge(base, amp, color, cover, soften):
        """A distant ridge. The skyline reads, and the edge stays hazy."""
        span = np.arange(WIDTH, dtype=np.float32)
        peaks = (
            base
            + amp * np.sin(span * 0.012 + base * 0.01)
            + amp * 0.55 * np.sin(span * 0.031 + 1.4)
            + amp * 0.22 * np.sin(span * 0.068 + 0.6)
        )
        depth = yy - peaks[None, :]
        mask = np.clip(depth / soften, 0, 1)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.0)
        crest = np.exp(-np.clip(depth, 0, None) / (soften * 0.65)) * mask
        lit = np.minimum(color + np.array([48, 40, 32], np.float32), 200)
        tone = color * (0.94 + 0.1 * grain)[..., None]
        tone = tone * (1 - crest[..., None] * 0.72) + lit * (crest[..., None] * 0.72)
        alpha = (mask * cover)[..., None]
        frame[:] = frame * (1 - alpha) + tone * alpha

    lay_ridge(232, 68, np.array([58, 54, 50], np.float32), 0.97, 8)
    lay_ridge(292, 48, np.array([32, 30, 28], np.float32), 0.98, 6)

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
    # The front of the cone is bright. The flanks turn away and go dark.
    light = np.clip(facing ** 0.62 * 0.8 + np.clip(0.36 - flank * 1.05, 0, 1) * 0.42, 0, 1)
    narrow = cv2.resize(grain, (WIDTH // 7, HEIGHT), interpolation=cv2.INTER_AREA)
    gullies = cv2.resize(narrow, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    gullies = cv2.GaussianBlur(gullies, (9, 9), 0)
    shadow = np.array([6, 5, 8], np.float32)
    lit = np.array([92, 108, 150], np.float32)
    rock = shadow * (1 - light[..., None]) + lit * light[..., None]
    rock *= (0.78 + 0.36 * gullies)[..., None]
    limb = np.clip((np.abs(flank) - 0.55) / 0.45, 0, 1)
    rock *= (1 - limb * 0.58)[..., None]
    trough = np.exp(-((xx - (apex_x - tt * 36)) ** 2) / (2 * (22 + tt * 34) ** 2))
    rock *= (1 - trough * 0.28)[..., None]
    for base_x, amount in ((80, 0.28), (220, -0.18), (360, 0.22), (620, 0.24), (760, -0.16), (900, 0.2)):
        x0, y0 = apex_x, apex_y + 16
        x1, y1 = float(base_x), BASE_Y + 8
        vx, vy = x1 - x0, y1 - y0
        length = float(np.hypot(vx, vy)) + 1e-6
        along = ((xx - x0) * vx + (yy - y0) * vy) / (length * length)
        dist = np.abs((xx - x0) * vy - (yy - y0) * vx) / length
        span_w = 18 + 48 * np.clip(along, 0, 1)
        fall = np.clip(1 - dist / (span_w + 1e-3), 0, 1) ** 2
        band = (along > 0.05) & (along < 1.02) & cone
        shade = np.where(band, fall, 0).astype(np.float32)
        if amount > 0:
            rock *= (1 - shade * amount)[..., None]
        else:
            rock += (shade * (-amount) * 42)[..., None]
    # A rounded nose in the middle, so that part sits in front of the flanks.
    belly = np.exp(-0.5 * ((tt - 0.42) / 0.36) ** 2)
    nose = np.exp(-0.5 * ((flank + 0.05) / 0.16) ** 2) * belly
    turn = np.exp(-0.5 * ((np.abs(flank) - 0.34) / 0.1) ** 2) * belly
    rock += nose[..., None] * np.array([16, 22, 34], np.float32)
    rock *= (1 - turn * 0.52)[..., None]
    frame[cone] = np.clip(rock, 0, 255)[cone]

    meadow = (yy >= 481) & ~cone
    coarse = cv2.resize(grain, (WIDTH // 28, HEIGHT // 28), interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(coarse, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    soil = np.array([20, 46, 24], np.float32)
    leaf = np.array([30, 74, 34], np.float32)
    patch = np.clip(coarse, 0, 1)
    mottle = np.clip(0.2 + grain * 0.8, 0, 1)
    shade = 1.0 - 0.22 * np.clip((yy - 481) / 58, 0, 1)
    tone = soil * (1 - patch[..., None]) + leaf * patch[..., None]
    tone *= (0.82 + 0.28 * mottle[..., None]) * shade[..., None]
    frame[meadow] = np.clip(tone, 0, 255)[meadow]
    return np.clip(frame, 0, 255).astype(np.uint8)


def sway_grass(frame, index):
    """Tips lean left with the clouds. A gust travels through the meadow."""
    gust = 0.65 + 0.35 * np.sin(index * 0.22)
    grass = np.random.default_rng(11)
    count = 1500
    xs = grass.integers(0, WIDTH, size=count)
    ys = grass.integers(484, HEIGHT - 1, size=count)
    heights = grass.integers(4, 22, size=count)
    leans = grass.integers(-4, 5, size=count)
    picks = grass.integers(0, 5, size=count)
    blades = (
        (24, 58, 26),
        (32, 82, 34),
        (18, 48, 22),
        (40, 98, 42),
        (28, 70, 30),
    )
    for blade in range(count):
        x = int(xs[blade])
        y = int(ys[blade])
        top = max(472, y - int(heights[blade]))
        reach = (y - top) / 22.0
        wave = np.sin(index * 0.22 + x * 0.02)
        shift = (-8.0 * gust + wave * 4.5) * reach + int(leans[blade]) * 0.35
        mid_y = (y + top) // 2
        mid_x = int(np.clip(x + shift * 0.35, 0, WIDTH - 1))
        tip_x = int(np.clip(x + shift, 0, WIDTH - 1))
        color = blades[int(picks[blade])]
        cv2.line(frame, (x, y), (mid_x, mid_y), color, 1, cv2.LINE_AA)
        cv2.line(frame, (mid_x, mid_y), (tip_x, top), color, 1, cv2.LINE_AA)


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
    """Soft cumulus banks. They drift from right to left and repeat."""
    color = np.zeros_like(frame, np.float32)
    cover = np.zeros((HEIGHT, WIDTH), np.float32)
    banks = (
        (120, 70, 0.4),
        (430, 48, 1.2),
        (760, 76, 2.1),
        (250, 150, 0.6),
        (640, 158, 1.7),
    )
    span = WIDTH + 320
    # One eighth of a pass per clip, so the drift stays smooth behind the lava.
    speed = span / (FRAMES * 8)
    for cx, cy, phase in banks:
        ox = (cx + 140 - index * speed) % span - 140
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
    sky = (frame[:, :, 0].astype(np.float32) - frame[:, :, 2].astype(np.float32) - 6) / 18.0
    sky = cv2.GaussianBlur(np.clip(sky, 0, 1), (5, 5), 1.2)
    alpha = np.clip(cover, 0, 0.78) * sky
    alpha = alpha[..., None]
    base = frame.astype(np.float32)
    frame[:] = np.clip(base * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)


def add_flames(frame, index, billow):
    """Lava above the crater gathers into clouds. Hotter lava rises into them."""
    cover = np.zeros((HEIGHT, WIDTH), np.float32)
    lit = np.zeros((HEIGHT, WIDTH), np.float32)
    heart = np.zeros((HEIGHT, WIDTH), np.float32)
    cx, cy = float(CRATER[0]), float(CRATER[1])
    for lobe in range(5):
        phase = lobe * 1.9
        lift = (index * 1.1 + lobe * 11.0) % 28.0
        y = cy - 22 - lobe * 18 - lift
        if y < 10:
            continue
        sample_y = int(np.clip(round(y), 0, HEIGHT - 1))
        sample_x = int(np.clip(round(cx), 0, WIDTH - 1))
        grain = float(billow[sample_y, sample_x])
        radius = (34 + 12 * grain) * (0.92 + 0.1 * np.sin(phase))
        amount = 1.0
        center_x = cx + np.sin(index * 0.16 + phase) * 14
        center = (int(round(center_x)), int(round(y)))
        cv2.circle(cover, center, max(8, int(round(radius))), amount, -1)
        cv2.circle(lit, center, max(6, int(round(radius * 0.9))), 0.95, -1)
        cv2.circle(heart, center, max(4, int(round(radius * 0.4))), 0.5, -1)
        for ox, oy, scale in (
            (0.42, -0.12, 0.78),
            (-0.4, 0.14, 0.74),
            (0.12, 0.32, 0.7),
            (-0.16, -0.28, 0.72),
        ):
            drift = np.sin(index * 0.22 + phase + ox) * 3
            spot = (int(round(center_x + ox * radius + drift)), int(round(y + oy * radius)))
            puff = max(5, int(round(radius * scale)))
            cv2.circle(cover, spot, puff, amount, -1)
            cv2.circle(lit, spot, max(4, int(round(puff * 0.88))), 0.9, -1)
            cv2.circle(heart, spot, max(3, int(round(puff * 0.32))), 0.4, -1)
    for n in range(5):
        phase = n * 1.4
        rise = (index * 2.4 + n * 24.0) % 130.0
        y = cy + 4 - rise
        if y < 8:
            continue
        fade = (1.0 - rise / 130.0) ** 0.75
        x = cx + np.sin(phase + index * 0.18) * (8 + rise * 0.08)
        sample_y = int(np.clip(round(y), 0, HEIGHT - 1))
        sample_x = int(np.clip(round(x), 0, WIDTH - 1))
        grain = float(billow[sample_y, sample_x])
        radius = (12 + 10 * fade) * (0.85 + 0.3 * grain)
        amount = 1.0 * fade
        center = (int(round(x)), int(round(y)))
        cv2.circle(heart, center, max(4, int(round(radius))), float(amount), -1)
        cv2.circle(lit, center, max(5, int(round(radius * 1.15))), 1.0, -1)
    cover = cv2.GaussianBlur(cover, (31, 31), 8.0)
    lit = cv2.GaussianBlur(lit, (21, 21), 5.0)
    heart = cv2.GaussianBlur(heart, (15, 15), 3.5)
    cover = np.clip(cover, 0, 1)
    lit = np.clip(lit, 0, 1) * cover
    heart = np.clip(heart, 0, 1) * cover
    crust = np.array([6, 12, 42], np.float32)
    mid = np.array([10, 40, 110], np.float32)
    core = np.array([96, 196, 255], np.float32)
    color = crust * (1 - lit)[..., None] + mid * lit[..., None]
    color = color * (1 - heart)[..., None] + core * heart[..., None]
    alpha = cover[..., None]
    base = frame.astype(np.float32)
    frame[:] = np.clip(base * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)


def cone_lip(x):
    """The cone surface under x. Pieces land here and then slide."""
    apex_x, apex_y = APEX
    span = BASE_Y - apex_y
    for y in range(int(apex_y), int(BASE_Y)):
        tt = min((y - apex_y) / span, 1.0)
        half = 78 + 360 * (tt ** 1.12)
        skirt = max(0.0, (tt - 0.7) / 0.3)
        half += min(skirt, 1.0) ** 2 * 120
        jag = 1 + 0.02 * np.sin(y * 0.05 + 0.4) + 0.012 * np.sin(y * 0.13)
        center = apex_x - tt * 18 + np.sin(tt * 3.1) * 6
        if abs(x - center) <= half * jag:
            return float(y)
    return BASE_Y


def stamp_melt(mask, hot, x, y, length, back_x, back_y, amount, salt):
    """A lumpy clot stretched along its path. The hot core sits in the head."""
    if length < 4 or amount <= 0:
        return
    norm = float(np.hypot(back_x, back_y))
    if norm < 1e-3:
        bx, by = 0.0, -1.0
    else:
        bx, by = back_x / norm, back_y / norm
    px, py = -by, bx
    tail_x = x + bx * length
    tail_y = y + by * length
    steps = 8

    def skin(t, side, phase):
        neck = 0.22 + 0.35 * abs(np.sin(salt))
        if 0.32 < t < 0.58:
            pinch = neck + (1 - neck) * abs(t - 0.45) / 0.13
        else:
            pinch = 0.55 + 0.45 * abs(np.sin(t * np.pi * (1.3 + abs(np.sin(salt)))))
        lump = 0.5 + 0.5 * abs(np.sin(t * 8.4 + salt * 2.1 + phase))
        wide = max(0.7, min(length * 0.17 * pinch * lump, length * 0.4))
        mx = tail_x + (x - tail_x) * t + px * np.sin(t * 6.4 + salt) * length * 0.04
        my = tail_y + (y - tail_y) * t + py * np.sin(t * 6.4 + salt) * length * 0.04
        return mx + px * wide * side, my + py * wide * side

    pts = [skin(step / steps, 1.0, 0.0) for step in range(steps + 1)]
    for step in range(1, 5):
        ang = step / 5 * np.pi
        lobe = 0.72 + 0.28 * np.sin(ang * 2.0 + salt)
        pts.append((
            x + np.cos(ang) * px * length * 0.1 * lobe - bx * np.sin(ang) * length * 0.07,
            y + np.cos(ang) * py * length * 0.1 * lobe - by * np.sin(ang) * length * 0.07,
        ))
    pts.extend(skin(step / steps, -1.0, 1.8) for step in range(steps, -1, -1))
    contour = np.array([(int(round(a)), int(round(b))) for a, b in pts], np.int32)
    cv2.fillPoly(mask, [contour], float(amount))
    core = [skin(0.58 + 0.42 * step / steps, 0.42, 0.4) for step in range(steps + 1)]
    core.extend(skin(0.58 + 0.42 * step / steps, -0.42, 2.1) for step in range(steps, -1, -1))
    cv2.fillPoly(hot, [np.array([(int(round(a)), int(round(b))) for a, b in core], np.int32)], float(amount))
    if amount > 0.55 and length > 12:
        fleck = length * (0.045 + 0.02 * abs(np.sin(salt)))
        fx = tail_x + bx * length * 0.18
        fy = tail_y + by * length * 0.18
        chip = np.array([
            (int(round(fx - px * fleck)), int(round(fy - py * fleck))),
            (int(round(fx + bx * fleck * 1.4)), int(round(fy + by * fleck * 1.4))),
            (int(round(fx + px * fleck * 0.6)), int(round(fy + py * fleck * 0.6))),
        ], np.int32)
        cv2.fillPoly(mask, [chip], float(amount * 0.4))


def add_fountain(frame, index):
    """Clots spill from just under the flame, fall down the face, then slide and cool."""
    cx, cy = float(CRATER[0]), float(CRATER[1])
    mask = np.zeros(frame.shape[:2], np.float32)
    drops = np.zeros(frame.shape[:2], np.float32)
    heads = np.zeros(frame.shape[:2], np.float32)
    pour = np.zeros(frame.shape[:2], np.float32)
    for n in range(8):
        phase = n * 2.399
        side = 1.0 if np.sin(phase * 1.3) >= 0 else -1.0
        lip = 10 + 36 * abs(np.sin(phase * 0.91))
        start_x = cx + side * lip
        start_y = cy + 8 + (n % 4) * 2.0
        drop = 110 + 70 * abs(np.sin(phase * 1.15))
        lean = side * (4 + 0.08 * drop)
        fall_len = 34 + (n * 3 % 10)
        slide_len = 84 + (n * 7 % 16)
        trail_len = 46
        life = (index * 0.9 + n * 11.0) % (fall_len + slide_len + trail_len)
        land_x = start_x + lean
        land_y = min(start_y + drop, BASE_Y - 36)
        hit = fall_len
        if life < hit:
            u = life / max(hit, 1.0)
            fall_u = u ** 1.15
            head_x = start_x + lean * fall_u
            head_y = start_y + (land_y - start_y) * fall_u
            span = max(head_y - start_y, 6.0)
            steps = max(6, int(span / 4.0))
            for step in range(steps):
                t = step / (steps - 1) if steps > 1 else 1.0
                bow = np.sin(t * np.pi) * side * (3 + 2 * np.sin(phase))
                x = start_x + (head_x - start_x) * t + bow
                y = start_y + (head_y - start_y) * t
                amount = 0.16 + 0.7 * (t ** 1.4)
                radius = 1.6 + 1.8 * t
                center = (int(round(x)), int(round(y)))
                cv2.circle(pour, center, max(2, int(round(radius))), float(amount), -1)
            length = 11 + 16 * fall_u
            stamp_melt(
                drops, heads, head_x, head_y, length,
                start_x - head_x, start_y - head_y - 5,
                0.95, phase,
            )
            continue
        if life < hit + slide_len:
            slide_u = (life - hit) / slide_len
            trail_fade = 1.0
            head_fade = 1.0
        else:
            slide_u = 1.0
            trail_u = (life - hit - slide_len) / trail_len
            if trail_u >= 1:
                continue
            trail_fade = 1 - trail_u
            head_fade = max(0.0, 1 - trail_u / 0.28)
        spill = side
        side = np.sign(land_x - APEX[0])
        if abs(land_x - APEX[0]) < 36:
            side = spill
        run = (BASE_Y - 18 - land_y) * 0.82

        def slope_at(frac):
            sx = land_x + side * frac * run * 0.62
            sy = max(land_y + frac * run, cone_lip(sx) + 4)
            return sx, sy

        for mark in range(7):
            frac = slide_u * mark / 7
            sx, sy = slope_at(frac)
            if sy >= BASE_Y - 10 or not (16 < sx < WIDTH - 16):
                continue
            behind = 1 - mark / 7
            length = 4.2 + 7.4 * (1 - behind) ** 1.4
            amount = 0.28 * (0.22 + 0.78 * (1 - behind)) * trail_fade
            if amount < 0.04:
                continue
            sx += np.sin(phase + mark * 1.3) * 2.2 * behind
            stamp_melt(drops, heads, sx, sy, length, -side * 0.62, -1.0, amount, phase + mark)
        if head_fade > 0.04:
            sx, sy = slope_at(slide_u)
            if sy < BASE_Y - 10 and 16 < sx < WIDTH - 16:
                stamp_melt(drops, heads, sx, sy, 15.5, -side * 0.62, -1.0, 0.9 * head_fade, phase + 2.0)
    pour = cv2.GaussianBlur(pour, (7, 7), 1.6)
    drops = np.maximum(drops, np.clip(pour, 0, 1))
    soft = cv2.GaussianBlur(mask, (11, 11), 2.8)
    core = cv2.GaussianBlur(mask, (21, 21), 5.5)
    drop_cover = cv2.GaussianBlur(drops, (3, 3), 0.6)
    drop_heart = cv2.GaussianBlur(heads, (3, 3), 0.6)
    crown = np.clip(soft * 1.45, 0, 1)
    # A dense crown hides the clouds, so the rising lava does not break into blocks.
    crown = np.maximum(crown, np.clip((crown - 0.42) / 0.28, 0, 1))
    cover = np.clip(np.maximum(crown, drop_cover * 1.2), 0, 1)
    heart = np.clip((core - 0.25) / 0.75, 0, 1) * np.clip(soft * 1.45, 0, 1)
    heart = np.clip(heart + drop_heart * 0.9, 0, 1)
    crust = np.array([14, 58, 150], np.float32)
    mid = np.array([26, 125, 235], np.float32)
    glow = np.array([110, 210, 255], np.float32)
    color = crust * (1 - heart)[..., None] + mid * heart[..., None]
    color = color * (1 - heart)[..., None] + glow * heart[..., None]
    alpha = cover[..., None]
    base = frame.astype(np.float32)
    frame[:] = np.clip(base * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)


def _unit(vector):
    norm = float(np.hypot(vector[0], vector[1]))
    if norm < 1e-4:
        return np.array([0.0, 1.0], np.float32)
    return np.asarray(vector, np.float32) / norm


def _bezier(p0, p1, p2, p3, count=14):
    t = np.linspace(0, 1, count, dtype=np.float32)[:, None]
    u = 1 - t
    pts = (u ** 3) * p0 + 3 * (u ** 2) * t * p1 + 3 * u * (t ** 2) * p2 + (t ** 3) * p3
    return pts


def _bezier_at(p0, p1, p2, p3, t):
    u = 1 - t
    return (u ** 3) * p0 + 3 * (u ** 2) * t * p1 + 3 * u * (t ** 2) * p2 + (t ** 3) * p3


def _bezier_tangent(p0, p1, p2, p3, t):
    u = 1 - t
    slope = 3 * (u ** 2) * (p1 - p0) + 6 * u * t * (p2 - p1) + 3 * (t ** 2) * (p3 - p2)
    return _unit(slope)


def _pull_back(points, pixels):
    direction = _unit(points[1] - points[0])
    extra = points[0] - direction * pixels
    return np.vstack([extra, points])


def lava_limbs(width, height):
    """A trunk that fans down the cone and stops where the slope meets the ground."""
    scale = np.array([width, height], np.float32)
    center_x = 0.49 * width
    low_y = BASE_Y - 2

    def at(x, y):
        return np.array([x, y], np.float32) * scale

    limbs = []

    def hold(points):
        """Keep each step downhill, and stop at the foot of the cone."""
        held = []
        for point in points:
            y = min(float(point[1]), low_y)
            x = float(point[0])
            if held and y < float(held[-1][1]) + 1.5:
                y = min(float(held[-1][1]) + 1.5, low_y)
            if held and float(held[-1][1]) >= low_y - 1.0:
                break
            held.append(np.array([x, y], np.float32))
        if not held:
            held.append(np.array([center_x, low_y], np.float32))
        while len(held) < 4:
            last = held[-1]
            held.append(np.array([float(last[0]), min(float(last[1]) + 0.6, low_y)], np.float32))
        return held[:4]

    def add(p0, p1, p2, p3, start_w, end_w, phase, pinch):
        held = hold((p0, p1, p2, p3))
        curve = _bezier(*held)
        if pinch or start_w < 20:
            curve = _pull_back(curve, 8 if start_w < 3.5 else 20)
        limbs.append((curve, start_w, end_w, phase, pinch))
        return held

    def limb_from(origin, tangent, normal, length, start_w, phase, pinch, end_ratio):
        side = float(normal[0])
        if abs(side) < 0.35:
            side = float(tangent[0])
        side = 1.0 if side >= 0 else -1.0
        droop = 1.05 + 0.22 * abs(float(np.sin(phase * 1.3)))
        reach = 0.48 + 0.28 * abs(float(np.sin(phase * 0.9)))
        direction = _unit(np.array([side * reach, droop], np.float32))
        room = max(22.0, low_y - 8 - float(origin[1]))
        travel = min(float(length), room / max(float(direction[1]), 0.3))
        bow = np.array([-side * 16.0, 14.0], np.float32)
        p0 = origin
        p1 = origin + direction * travel * 0.32 + tangent * 4
        p2 = origin + direction * travel * 0.7 + bow
        p3 = origin + direction * travel
        held = add(p0, p1, p2, p3, start_w, max(0.45, start_w * end_ratio), phase, pinch)
        return held[0], held[1], held[2], held[3], direction

    def side_twig(p0, p1, p2, p3, t, length, parent_w, phase, end_ratio, side=None):
        origin = _bezier_at(p0, p1, p2, p3, t)
        tangent = _bezier_tangent(p0, p1, p2, p3, t)
        normal = np.array([-tangent[1], tangent[0]], np.float32)
        if side is None:
            if (origin[0] - center_x) * normal[0] < 0:
                normal = -normal
        elif side * normal[0] < 0:
            normal = -normal
        limb_from(origin, tangent, normal, length, max(0.9, parent_w), phase, True, end_ratio)

    def grow(parent, t, length, start_w, phase, side, fine):
        p0, p1, p2, p3 = parent
        origin = _bezier_at(p0, p1, p2, p3, t)
        tangent = _bezier_tangent(p0, p1, p2, p3, t)
        normal = np.array([-tangent[1], tangent[0]], np.float32)
        if side * normal[0] < 0:
            normal = -normal
        branch = limb_from(origin, tangent, normal, length, start_w, phase, False, 0.34 if fine else 0.5)
        spots = (0.42, 0.74) if fine else (0.58,)
        for slot, spot in enumerate(spots):
            side_twig(
                *branch[:4],
                spot,
                (78 if fine else 96) - slot * 18,
                start_w * (0.32 if fine else 0.4),
                phase + slot * 0.7,
                0.22 if fine else 0.3,
            )
        end = branch[3]
        if float(end[1]) >= low_y - 16:
            return
        direction = branch[4]
        # Near the foot the last forks open wide. Higher up they keep falling.
        low = float(end[1]) > low_y - 150
        if low:
            forks = ((-0.45, 0.85, 0.38), (0.65, 0.7, 0.3), (1.45, 0.8, 0.22))
            down = 0.72
        elif fine:
            forks = ((0.22, 0.62, 0.48), (0.85, 0.4, 0.26))
            down = 0.95
        else:
            forks = ((0.28, 0.7, 0.5),)
            down = 0.95
        for slot, (spread, keep, width_ratio) in enumerate(forks):
            tip = _unit(np.array([
                direction[0] * 0.4 + normal[0] * spread,
                down,
            ], np.float32))
            room = max(18.0, low_y - 8 - float(end[1]))
            tip_len = min(length * keep, room / max(float(tip[1]), 0.3))
            tip_w = max(0.7, start_w * width_ratio)
            end_ratio = 1.7 if low else 0.4
            c0 = end
            c1 = end + tip * tip_len * 0.34
            c2 = end + tip * tip_len * 0.7 + np.array([-normal[0] * 3.0, 8.0], np.float32)
            c3 = end + tip * tip_len
            add(c0, c1, c2, c3, tip_w, max(0.4, tip_w * end_ratio), phase + slot * 0.6, not low)

    trunk = (at(0.518, 0.348), at(0.508, 0.50), at(0.496, 0.68), at(0.49, 0.84))
    add(*trunk, 14, 10, 0.2, False)
    grow(trunk, 0.14, 240, 7.0, 0.3, -1, False)
    grow(trunk, 0.24, 190, 5.0, 1.35, 1, False)
    grow(trunk, 0.38, 220, 4.6, 0.55, -1, True)
    grow(trunk, 0.5, 160, 3.2, 1.9, 1, True)
    grow(trunk, 0.62, 200, 3.4, 0.85, -1, True)
    grow(trunk, 0.72, 145, 2.6, 2.4, 1, True)
    grow(trunk, 0.82, 130, 2.2, 1.25, -1, True)
    grow(trunk, 0.9, 100, 1.7, 0.45, 1, True)
    side_twig(*trunk, 0.44, 110, 2.2, 0.8, 0.2, -1)
    side_twig(*trunk, 0.78, 90, 1.8, 1.6, 0.16, 1)

    def foot_run(mark, side, phase, length, width):
        """A thin channel that opens across the lower slope and stops at the foot."""
        origin = _bezier_at(*trunk, mark)
        droop = 0.92 + 0.2 * abs(float(np.sin(phase)))
        reach = 0.7 + 0.38 * abs(float(np.sin(phase * 1.4)))
        direction = _unit(np.array([side * reach, droop], np.float32))
        room = max(16.0, low_y - 10 - float(origin[1]))
        travel = min(float(length), room / max(float(direction[1]), 0.28))
        bow = np.array([side * 22.0, 16.0], np.float32)
        p0 = origin
        p1 = origin + direction * travel * 0.3 + np.array([side * 14.0, 8.0], np.float32)
        p2 = origin + direction * travel * 0.68 + bow
        p3 = origin + direction * travel
        add(p0, p1, p2, p3, width, width * 4.2, phase, False)

    for mark, side, phase, length, width in (
        (0.58, -1, 0.2, 280, 2.2),
        (0.58, 1, 1.3, 230, 1.7),
        (0.7, -1, 0.7, 250, 1.6),
        (0.7, 1, 1.8, 300, 2.0),
        (0.8, -1, 1.4, 240, 1.4),
        (0.8, 1, 0.45, 210, 1.3),
        (0.9, -1, 2.1, 180, 1.1),
        (0.9, 1, 0.95, 220, 1.3),
    ):
        foot_run(mark, side, phase, length, width)
    return limbs


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


def drift_x(y, index):
    """Sideways drift shared by every limb at the same height, so forks stay joined."""
    drift = index * 0.02
    return np.sin(y * 0.045 + drift) * 4 + np.sin(y * 0.095 + 1.1) * 1.6


def meander(pos, index):
    """Bend every limb by the same sideways drift so forks stay joined. Y only increases."""
    out = pos.copy()
    for i in range(len(out)):
        out[i, 0] += drift_x(float(out[i, 1]), index)
    for i in range(1, len(out)):
        floor = BASE_Y - 2
        if out[i, 1] < out[i - 1, 1] + 0.4:
            out[i, 1] = min(out[i - 1, 1] + 0.4, floor)
        if out[i, 1] > floor:
            out[i, 1] = floor
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


def flow_wave(y, index):
    """A long soft band that creeps downhill. The same height shares one wave, so forks split it."""
    return float(np.sin((float(y) - index * 0.75) * 0.02))


def downhill_gain(y):
    """Streams stay narrow under the crater and open as they run down the slope."""
    down = np.clip((float(y) - 250.0) / (BASE_Y - 250.0), 0.0, 1.0)
    return 1.0 + 1.7 * (down ** 1.15)


def limb_strip(pos, progress, tangent, start_w, end_w, phase, pinch, inset, shift, index):
    """A smooth limb. Shift moves the lit side toward the upper light."""
    left = []
    right = []
    light = np.array([-0.45, -0.9], np.float32)
    for i in range(len(pos)):
        t = float(progress[i])
        ease = t * t * (3 - 2 * t)
        width = start_w + (end_w - start_w) * ease
        if pinch and t > 0.7 and end_w < start_w:
            width *= (1.0 - t) / 0.3
        wave = flow_wave(pos[i, 1], index)
        lump = 0.5 + 0.5 * np.sin(t * 6.2 + phase + float(pos[i, 1]) * 0.02)
        grow = 0.78 + 0.22 * max(0.0, wave) if end_w > start_w else 1.0
        width *= (0.8 + 0.28 * lump + 0.12 * max(0.0, wave)) * grow * downhill_gain(pos[i, 1])
        width = min(width, 6.0 + 9.0 * np.clip((float(pos[i, 1]) - 250.0) / (BASE_Y - 250.0), 0, 1))
        half = max(0.45, width * inset)
        nx = -float(tangent[i, 1])
        ny = float(tangent[i, 0])
        if nx * light[0] + ny * light[1] < 0:
            nx, ny = -nx, -ny
        cx = pos[i, 0] + nx * half * shift
        cy = pos[i, 1] + ny * half * shift
        px = -float(tangent[i, 1])
        py = float(tangent[i, 0])
        left.append((cx + px * half, cy + py * half))
        right.append((cx - px * half, cy - py * half))
    if len(left) < 2:
        return None
    return np.array(left + right[::-1], np.int32)


def paint_flow(glow, pos, progress, tangent, start_w, end_w, pinch, index):
    """A soft heat gradient along the limb, so the lava reads as one slow liquid."""
    for i in range(len(pos) - 1):
        corners = []
        heats = []
        for k in (i, i + 1):
            t = float(progress[k])
            ease = t * t * (3 - 2 * t)
            width = start_w + (end_w - start_w) * ease
            if pinch and t > 0.7 and end_w < start_w:
                width *= (1.0 - t) / 0.3
            wave = max(0.0, flow_wave(pos[k, 1], index))
            if end_w > start_w:
                width *= 0.78 + 0.22 * wave
            width *= downhill_gain(pos[k, 1])
            width = min(width, 6.0 + 9.0 * np.clip((float(pos[k, 1]) - 250.0) / (BASE_Y - 250.0), 0, 1))
            heat = wave * wave * (3 - 2 * wave)
            half = max(0.3, width * (0.18 + 0.28 * heat))
            px = -float(tangent[k, 1])
            py = float(tangent[k, 0])
            corners.append((pos[k, 0] + px * half, pos[k, 1] + py * half))
            corners.append((pos[k, 0] - px * half, pos[k, 1] - py * half))
            heats.append(heat)
        amount = 0.5 * (heats[0] + heats[1])
        if amount < 0.34:
            continue
        quad = np.array([corners[0], corners[2], corners[3], corners[1]], np.int32)
        cv2.fillPoly(glow, [quad], int(amount * 255))


def paint_channels(frame, index):
    """Slow lava leaves the crater, spreads downhill, and grows wider as it descends."""
    body = np.zeros(frame.shape[:2], np.uint8)
    lit = np.zeros(frame.shape[:2], np.uint8)
    glow = np.zeros(frame.shape[:2], np.uint8)
    for points, start_w, end_w, phase, pinch in lava_limbs(WIDTH, HEIGHT):
        pos, progress, _tangent = path_samples(points, 42)
        pos = meander(pos, index)
        tangent = tangents_of(pos)
        limb = limb_strip(pos, progress, tangent, start_w, end_w, phase, pinch, 1.0, 0.0, index)
        shine = limb_strip(pos, progress, tangent, start_w, end_w, phase, pinch, 0.34, 0.4, index)
        if limb is not None:
            cv2.fillPoly(body, [limb], 255)
        if shine is not None:
            cv2.fillPoly(lit, [shine], 255)
        paint_flow(glow, pos, progress, tangent, start_w, end_w, pinch, index)
    cover = cv2.GaussianBlur(body, (5, 5), 1.05).astype(np.float32) / 255
    light = cv2.GaussianBlur(lit, (5, 5), 1.0).astype(np.float32) / 255
    heart = cv2.GaussianBlur(glow, (15, 15), 3.0).astype(np.float32) / 255
    cover = np.clip(cover, 0, 1)
    light = np.clip(light, 0, 1) * cover
    heart = np.clip(heart, 0, 1) * cover
    crust = np.array([6, 12, 42], np.float32)
    mid = np.array([10, 40, 110], np.float32)
    core = np.array([96, 196, 255], np.float32)
    color = crust * (1 - light)[..., None] + mid * light[..., None]
    color = color * (1 - heart)[..., None] + core * heart[..., None]
    alpha = cover[..., None]
    base = frame.astype(np.float32)
    frame[:] = np.clip(base * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)
    heat = np.clip(cover * 0.45 + light * 0.4 + heart * 0.7, 0, 1)
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
    add_crater(frame, index)
    melt = paint_channels(frame, index)
    billow = scrolled(lobe_field, index, 1.4, HEIGHT)
    add_flames(frame, index, billow)
    add_fountain(frame, index)
    sway_grass(frame, index)
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
