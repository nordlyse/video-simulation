"""Local camera app. It stores one frame for the Kornia Compose service."""

import ctypes
import sys
import time
from pathlib import Path

import AVFoundation
import CoreMedia
import Quartz
import objc
from AppKit import NSBitmapImageFileTypePNG, NSBitmapImageRep
from Foundation import NSDate, NSObject, NSRunLoop

# CoreVideo pixel format BGRA.
PIXEL_FORMAT_BGRA = 1111970369
WARM_FRAMES = 15


def wait_until(ready, seconds):
    loop = NSRunLoop.currentRunLoop()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if ready():
            return True
        loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))
    return ready()


class FrameSink(NSObject):
    def initWithPath_(self, path):
        self = objc.super(FrameSink, self).init()
        if self is None:
            return None
        self.path = path
        self.seen = 0
        self.kept = False
        self.failure = None
        return self

    def captureOutput_didOutputSampleBuffer_fromConnection_(self, _output, sampleBuffer, _connection):
        if self.kept:
            return
        self.seen += 1
        if self.seen < WARM_FRAMES:
            return
        pixel = CoreMedia.CMSampleBufferGetImageBuffer(sampleBuffer)
        try:
            store_png(pixel, self.path)
        except Exception as error:
            self.failure = str(error)
        self.kept = True


def store_png(pixel, path):
    image = Quartz.CIImage.imageWithCVPixelBuffer_(pixel)
    context = Quartz.CIContext.contextWithOptions_(None)
    rendered = context.createCGImage_fromRect_(image, image.extent())
    if rendered is None:
        raise RuntimeError("The camera frame could not be rendered.")
    rep = NSBitmapImageRep.alloc().initWithCGImage_(rendered)
    data = rep.representationUsingType_properties_(NSBitmapImageFileTypePNG, None)
    if data is None or not data.writeToFile_atomically_(path, True):
        raise RuntimeError("The camera frame could not be written.")
    if not Path(path).is_file():
        raise RuntimeError("The camera frame file is missing after writing.")


def frame_queue():
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    library.dispatch_queue_create.restype = ctypes.c_void_p
    library.dispatch_queue_create.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    raw = library.dispatch_queue_create(b"camera.frames", None)
    return objc.objc_object(c_void_p=ctypes.c_void_p(raw))


def hold_frame(path):
    print("Waiting for camera access. Allow it if macOS asks.", file=sys.stderr)
    state = {"granted": None}

    def on_access(answer):
        state["granted"] = bool(answer)

    AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        AVFoundation.AVMediaTypeVideo,
        on_access,
    )
    if not wait_until(lambda: state["granted"] is not None, 120) or not state["granted"]:
        print("Camera access is not granted.", file=sys.stderr)
        return 1

    device = AVFoundation.AVCaptureDevice.defaultDeviceWithMediaType_(AVFoundation.AVMediaTypeVideo)
    if device is None:
        print("No camera is available.", file=sys.stderr)
        return 1

    session = AVFoundation.AVCaptureSession.alloc().init()
    session.beginConfiguration()
    if session.canSetSessionPreset_(AVFoundation.AVCaptureSessionPreset1280x720):
        session.setSessionPreset_(AVFoundation.AVCaptureSessionPreset1280x720)
    elif session.canSetSessionPreset_(AVFoundation.AVCaptureSessionPresetHigh):
        session.setSessionPreset_(AVFoundation.AVCaptureSessionPresetHigh)

    camera_input, error = AVFoundation.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
    if camera_input is None or not session.canAddInput_(camera_input):
        print(error or "The camera input could not be added.", file=sys.stderr)
        return 1
    session.addInput_(camera_input)

    video = AVFoundation.AVCaptureVideoDataOutput.alloc().init()
    video.setAlwaysDiscardsLateVideoFrames_(True)
    video.setVideoSettings_({"PixelFormatType": PIXEL_FORMAT_BGRA})
    if not session.canAddOutput_(video):
        print("The camera output could not be added.", file=sys.stderr)
        return 1
    session.addOutput_(video)
    session.commitConfiguration()

    sink = FrameSink.alloc().initWithPath_(path)
    video.setSampleBufferDelegate_queue_(sink, frame_queue())
    session.startRunning()
    arrived = wait_until(lambda: sink.kept, 8)
    session.stopRunning()

    if sink.failure:
        print(sink.failure, file=sys.stderr)
        return 1
    if not arrived:
        print("No frame arrived from the camera.", file=sys.stderr)
        return 1
    print(path)
    return 0


def main():
    if len(sys.argv) != 2:
        print("Usage: camera_app.py output.png", file=sys.stderr)
        return 64
    output = Path(sys.argv[1]).resolve()
    if not output.parent.is_dir():
        print(f"Missing folder: {output.parent}", file=sys.stderr)
        return 1
    return hold_frame(str(output))


if __name__ == "__main__":
    raise SystemExit(main())
