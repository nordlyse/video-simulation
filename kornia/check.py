"""Run a small Kornia image step so the container can show it works."""

import kornia
import torch

camera = torch.rand(1, 3, 32, 32)
clip = torch.rand(1, 3, 32, 32)
mixed = camera * 0.5 + clip * 0.5
gray = kornia.color.rgb_to_grayscale(mixed)
resized = kornia.geometry.transform.resize(mixed, (16, 16))

print(f"kornia {kornia.__version__}")
print(f"camera {tuple(camera.shape)}")
print(f"mixed {tuple(mixed.shape)}")
print(f"gray {tuple(gray.shape)}")
print(f"resized {tuple(resized.shape)}")
