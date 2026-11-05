# Video Simulation

This application uses [Kornia](https://github.com/kornia/kornia) to take the image from a camera, combine it with selected videos, and turn the result into a simulation of being in virtual places. The combined image is then sent to a virtual camera.

## Licences

These are the tools in the Compose image and the licences recorded with the installed packages.

| Tool | Version | Licence |
| --- | --- | --- |
| [Kornia](https://github.com/kornia/kornia) | 0.9.0 | Apache-2.0 |
| kornia_rs | 0.2.0 | Apache-2.0 |
| torch | 2.15.0+cpu | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| filelock | 4.2.4 | MIT |
| fsspec | 2026.9.0 | BSD-3-Clause |
| Jinja2 | 3.1.6 | BSD-3-Clause |
| MarkupSafe | 3.0.3 | BSD-3-Clause |
| mpmath | 1.3.0 | BSD-3-Clause |
| networkx | 3.7 | BSD-3-Clause |
| setuptools | 78.1.0 | MIT |
| sympy | 1.14.0 | BSD-3-Clause |
| typing_extensions | 4.16.0 | PSF-2.0 |
| pip | 25.0.1 | MIT |
| Python | 3.12 | PSF-2.0 |

Kornia requires torch and numpy. The torch install also brings in filelock, fsspec, Jinja2, MarkupSafe, mpmath, networkx, setuptools, sympy, and typing_extensions.

The PyTorch project text is BSD-3-Clause. The CPU wheel metadata adds the other licences in its row. NumPy’s own text is BSD-3-Clause, and the wheel also includes 0BSD, MIT, Zlib, and CC0-1.0 code. setuptools is MIT, and a vendored autocommand copy in that package is LGPL-3.0. sympy is BSD-3-Clause, and its bundled latex2sympy parser is MIT.

The image base is `python:3.12-slim`. Python is PSF-2.0. pip, which ships with that image, is MIT.

Kornia keeps only the person from the camera frame. MediaPipe 1.2.2 and its selfie segmenter model are Apache-2.0. The image also installs opencv-python-headless 5.0.0.93, absl-py 2.5.1 and flatbuffers 25.12.19, all Apache-2.0. MediaPipe imports certifi 2026.10.23, which is MPL-2.0, and matplotlib. If the frame has no person, the stored frame is empty.

The erupting volcano is a vector scene. A cone with ridges and gullies stands in front of hazy ridges, under darkened clouds. Ash billows out of the crater, and lava leaves the vent in broken channels that run downhill and split into fingers on the plain. The extracted person stands in the foreground and is shaded from that lava: the side toward the flow is warmer and the far side is darker. The result is `frames/ready/volcano.mp4`.

The local camera app is separate from Compose. It uses PyObjC 13.0, which is MIT:

| Tool | Version | Licence |
| --- | --- | --- |
| pyobjc-core | 13.0 | MIT |
| pyobjc-framework-AVFoundation | 13.0 | MIT |
| pyobjc-framework-Cocoa | 13.0 | MIT |
| pyobjc-framework-CoreAudio | 13.0 | MIT |
| pyobjc-framework-CoreMedia | 13.0 | MIT |
| pyobjc-framework-Quartz | 13.0 | MIT |
| Python | 3.14 | PSF-2.0 |
