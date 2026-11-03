"""Hold one camera frame with Kornia for a later mix with other videos."""

from pathlib import Path

import kornia
import torch

inbox = Path("/frames/inbox/live.png")
ready = Path("/frames/ready")

if not inbox.is_file():
    raise SystemExit(f"Missing camera frame: {inbox}")

frame = kornia.io.load_image(inbox, kornia.io.ImageLoadType.RGB32)
if frame.ndim != 3 or frame.shape[0] != 3:
    raise SystemExit(f"Unexpected frame shape {tuple(frame.shape)}")

ready.mkdir(parents=True, exist_ok=True)
stored = frame.unsqueeze(0).contiguous()
torch.save(stored, ready / "live.pt")

preview = (frame.clamp(0, 1) * 255).round().to(torch.uint8)
kornia.io.write_image(ready / "live.png", preview)

print(f"kornia {kornia.__version__}")
print(f"frame {tuple(stored.shape)} dtype {stored.dtype}")
print(f"min {float(stored.min()):.4f} max {float(stored.max()):.4f}")
print("ready /frames/ready/live.pt")
