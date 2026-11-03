"""Local camera app. A window shows the camera and holds one frame for Kornia."""

import ctypes
import sys
import threading
from pathlib import Path

import AVFoundation
import CoreMedia
import Quartz
import objc
from AppKit import (
    NSApp,
    NSApplication,
    NSApplicationActivationPolicyRegular,
    NSBackingStoreBuffered,
    NSButton,
    NSEvent,
    NSScreen,
    NSTextField,
    NSView,
    NSWindow,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable,
    NSWindowStyleMaskResizable,
    NSWindowStyleMaskTitled,
)
from Foundation import NSMakeRect, NSObject, NSPointInRect

# CoreVideo pixel format BGRA.
PIXEL_FORMAT_BGRA = 1111970369
WARM_FRAMES = 15


def store_png(pixel, path):
    image = Quartz.CIImage.imageWithCVPixelBuffer_(pixel)
    context = Quartz.CIContext.contextWithOptions_(None)
    rendered = context.createCGImage_fromRect_(image, image.extent())
    if rendered is None:
        raise RuntimeError("The camera frame could not be rendered.")
    from AppKit import NSBitmapImageFileTypePNG, NSBitmapImageRep

    rep = NSBitmapImageRep.alloc().initWithCGImage_(rendered)
    data = rep.representationUsingType_properties_(NSBitmapImageFileTypePNG, None)
    if data is None or not data.writeToFile_atomically_(path, True):
        raise RuntimeError("The camera frame could not be written.")
    if not Path(path).is_file():
        raise RuntimeError("The camera frame file is missing after writing.")


def dispatch_queue(name):
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    library.dispatch_queue_create.restype = ctypes.c_void_p
    library.dispatch_queue_create.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    raw = library.dispatch_queue_create(name, None)
    return objc.objc_object(c_void_p=ctypes.c_void_p(raw))


class FrameSink(NSObject):
    def initWithPath_owner_(self, path, owner):
        self = objc.super(FrameSink, self).init()
        if self is None:
            return None
        self.path = path
        self.owner = owner
        self.seen = 0
        self.arm = False
        self.failure = None
        return self

    def captureOutput_didOutputSampleBuffer_fromConnection_(self, _output, sampleBuffer, _connection):
        if not self.arm or self.seen < WARM_FRAMES:
            self.seen += 1
            return
        self.arm = False
        pixel = CoreMedia.CMSampleBufferGetImageBuffer(sampleBuffer)
        self.failure = None
        try:
            store_png(pixel, self.path)
        except Exception as error:
            self.failure = str(error)
        self.owner.performSelectorOnMainThread_withObject_waitUntilDone_("showHeld:", None, False)


class CameraWindow(NSObject):
    def initWithPath_(self, path):
        self = objc.super(CameraWindow, self).init()
        if self is None:
            return None
        self.path = path
        self.session = None
        self.preview = None
        self.sink = FrameSink.alloc().initWithPath_owner_(path, self)
        self.build_window()
        return self

    def build_window(self):
        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskMiniaturizable
            | NSWindowStyleMaskResizable
        )
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 960, 640),
            style,
            NSBackingStoreBuffered,
            False,
        )
        self.window.setTitle_("Camera")
        self.window.setDelegate_(self)

        self.preview_view = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 10, 10))
        self.button = NSButton.alloc().initWithFrame_(NSMakeRect(0, 0, 10, 10))
        self.button.setTitle_("Hold frame")
        self.button.setBezelStyle_(1)
        self.button.setTarget_(self)
        self.button.setAction_("holdFrame:")
        self.button.setEnabled_(False)

        self.status = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 10, 10))
        self.status.setStringValue_("Waiting for camera access.")
        self.status.setBezeled_(False)
        self.status.setDrawsBackground_(False)
        self.status.setEditable_(False)
        self.status.setSelectable_(False)

        content = self.window.contentView()
        content.addSubview_(self.preview_view)
        content.addSubview_(self.button)
        content.addSubview_(self.status)
        self.place_on_active_screen()
        self.layout_views()

    def place_on_active_screen(self):
        screen = NSScreen.mainScreen()
        mouse = NSEvent.mouseLocation()
        for candidate in NSScreen.screens():
            if NSPointInRect(mouse, candidate.frame()):
                screen = candidate
                break
        visible = screen.visibleFrame()
        width, height = 960, 640
        x = visible.origin.x + max(0, (visible.size.width - width) / 2)
        y = visible.origin.y + max(0, (visible.size.height - height) / 2)
        self.window.setFrame_display_(NSMakeRect(x, y, width, height), True)

    def layout_views(self):
        if not hasattr(self, "preview_view"):
            return
        bounds = self.window.contentView().bounds()
        width = bounds.size.width
        height = bounds.size.height
        self.preview_view.setFrame_(NSMakeRect(0, 52, width, max(1, height - 52)))
        self.button.setFrame_(NSMakeRect(16, 12, 140, 32))
        self.status.setFrame_(NSMakeRect(168, 16, max(1, width - 184), 24))
        if self.preview is not None:
            self.preview.setFrame_(self.preview_view.bounds())

    def windowDidResize_(self, _notification):
        self.layout_views()

    def applicationWillTerminate_(self, _notification):
        self.stop_session()

    def windowWillClose_(self, _notification):
        self.stop_session()
        NSApp.terminate_(None)

    def stop_session(self):
        session = self.session
        if session is not None and session.isRunning():
            session.stopRunning()

    def open(self):
        self.window.makeKeyAndOrderFront_(None)
        print("Camera window is open. Close it to quit.", file=sys.stderr)
        self.ask_access()

    def ask_access(self):
        status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(AVFoundation.AVMediaTypeVideo)
        if status == AVFoundation.AVAuthorizationStatusAuthorized:
            self.granted = True
            self.begin_session()
            return
        if status == AVFoundation.AVAuthorizationStatusDenied or status == AVFoundation.AVAuthorizationStatusRestricted:
            self.granted = False
            self.accessAnswer_(None)
            return
        print("Waiting for camera access. Allow it if macOS asks.", file=sys.stderr)

        def on_access(answer):
            self.granted = bool(answer)
            self.performSelectorOnMainThread_withObject_waitUntilDone_("accessAnswer:", None, False)

        AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVFoundation.AVMediaTypeVideo,
            on_access,
        )

    def accessAnswer_(self, _sender):
        if not getattr(self, "granted", False):
            self.status.setStringValue_("Camera access is not granted.")
            print("Camera access is not granted.", file=sys.stderr)
            return
        self.begin_session()

    def begin_session(self):
        device = AVFoundation.AVCaptureDevice.defaultDeviceWithMediaType_(AVFoundation.AVMediaTypeVideo)
        if device is None:
            self.status.setStringValue_("No camera is available.")
            print("No camera is available.", file=sys.stderr)
            return

        session = AVFoundation.AVCaptureSession.alloc().init()
        session.beginConfiguration()
        if session.canSetSessionPreset_(AVFoundation.AVCaptureSessionPreset1280x720):
            session.setSessionPreset_(AVFoundation.AVCaptureSessionPreset1280x720)
        elif session.canSetSessionPreset_(AVFoundation.AVCaptureSessionPresetHigh):
            session.setSessionPreset_(AVFoundation.AVCaptureSessionPresetHigh)

        camera_input, error = AVFoundation.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
        if camera_input is None or not session.canAddInput_(camera_input):
            message = str(error) if error is not None else "The camera input could not be added."
            self.status.setStringValue_(message)
            print(message, file=sys.stderr)
            return
        session.addInput_(camera_input)

        video = AVFoundation.AVCaptureVideoDataOutput.alloc().init()
        video.setAlwaysDiscardsLateVideoFrames_(True)
        video.setVideoSettings_({"PixelFormatType": PIXEL_FORMAT_BGRA})
        if not session.canAddOutput_(video):
            self.status.setStringValue_("The camera output could not be added.")
            print("The camera output could not be added.", file=sys.stderr)
            return
        session.addOutput_(video)
        video.setSampleBufferDelegate_queue_(self.sink, dispatch_queue(b"camera.frames"))
        session.commitConfiguration()

        preview = AVFoundation.AVCaptureVideoPreviewLayer.layerWithSession_(session)
        preview.setVideoGravity_(AVFoundation.AVLayerVideoGravityResizeAspect)
        self.preview_view.setWantsLayer_(True)
        self.preview_view.setLayer_(preview)
        self.preview = preview
        self.session = session
        self.layout_views()
        self.button.setEnabled_(True)
        self.status.setStringValue_("Holding the first frame.")
        self.sink.arm = True
        threading.Thread(target=session.startRunning, daemon=True).start()

    def holdFrame_(self, _sender):
        self.sink.failure = None
        self.status.setStringValue_("Holding the frame.")
        self.sink.arm = True

    def showHeld_(self, _sender):
        if self.sink.failure:
            self.status.setStringValue_(self.sink.failure)
            print(self.sink.failure, file=sys.stderr)
            return
        self.status.setStringValue_(self.sink.path)
        print(self.sink.path, flush=True)


def main():
    if len(sys.argv) != 2:
        print("Usage: camera_app.py output.png", file=sys.stderr)
        return 64
    output = Path(sys.argv[1]).resolve()
    if not output.parent.is_dir():
        print(f"Missing folder: {output.parent}", file=sys.stderr)
        return 1

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    window = CameraWindow.alloc().initWithPath_(str(output))
    app.setDelegate_(window)
    window.open()
    app.activateIgnoringOtherApps_(True)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
