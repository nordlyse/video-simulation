"""Keep only the person in one camera frame for a later video mix."""

from pathlib import Path

import kornia
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import ImageSegmenter, ImageSegmenterOptions

inbox = Path("/frames/inbox/live.png")
ready = Path("/frames/ready")
model_path = Path("/app/selfie_segmenter.tflite")
# A real person covers more of the frame than mask noise.
MIN_PERSON = 0.005


def person_mask(rgb):
    """Return a float mask in the shape of the frame. 1 is the person."""
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        output_confidence_masks=True,
    )
    with ImageSegmenter.create_from_options(options) as segmenter:
        result = segmenter.segment(image)

    masks = result.confidence_masks or []
    if not masks:
        raise RuntimeError("The segmenter returned no person mask.")
    # This selfie model returns one confidence mask for the person.
    mask = np.asarray(masks[0].numpy_view(), dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[:, :, 0] if mask.shape[2] == 1 else mask[0]
    return torch.from_numpy(np.array(mask, copy=True))


def store(frame):
    ready.mkdir(parents=True, exist_ok=True)
    stored = frame.unsqueeze(0).contiguous()
    torch.save(stored, ready / "live.pt")
    preview = (frame.clamp(0, 1) * 255).round().to(torch.uint8)
    kornia.io.write_image(ready / "live.png", preview)
    return stored


def main():
    if not inbox.is_file():
        raise SystemExit(f"Missing camera frame: {inbox}")

    frame = kornia.io.load_image(inbox, kornia.io.ImageLoadType.RGB32)
    if frame.ndim != 3 or frame.shape[0] != 3:
        raise SystemExit(f"Unexpected frame shape {tuple(frame.shape)}")

    rgb = (frame.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    mask = person_mask(rgb)
    mask = mask.to(dtype=torch.float32)
    if tuple(mask.shape) != tuple(frame.shape[-2:]):
        mask = kornia.geometry.transform.resize(
            mask.unsqueeze(0).unsqueeze(0),
            frame.shape[-2:],
            interpolation="bilinear",
        ).squeeze(0).squeeze(0)
    person = mask > 0.5
    coverage = float(person.float().mean())
    height, width = frame.shape[-2:]

    if coverage < MIN_PERSON:
        stored = store(torch.zeros(4, height, width))
        print(f"kornia {kornia.__version__}")
        print("person absent")
        print(f"frame {tuple(stored.shape)} dtype {stored.dtype}")
        print("ready /frames/ready/live.pt")
        return 0

    alpha = mask.clamp(0, 1).unsqueeze(0)
    rgba = torch.cat([frame, alpha], dim=0)
    stored = store(rgba)
    print(f"kornia {kornia.__version__}")
    print(f"person present coverage {coverage:.4f}")
    print(f"frame {tuple(stored.shape)} dtype {stored.dtype}")
    print(f"min {float(stored.min()):.4f} max {float(stored.max()):.4f}")
    print("ready /frames/ready/live.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
