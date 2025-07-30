
import cv2

GST_PIPELINE_STREAM = (
    "appsrc ! videoconvert ! "
    "x264enc tune=zerolatency bitrate=800 speed-preset=superfast ! "
    "mpegtsmux ! "
    "tcpserversink host=0.0.0.0 port=5000"
)

GST_PIPELINE_CAMERA = (
    "nvarguscamerasrc sensor-mode=3 ! "
    "video/x-raw(memory:NVMM), width={width}, height={height}, exposuretimerange=(string)2000000 2000000, framerate={fps}/1, format=NV12 ! "
    "nvvidconv flip-method={flip} ! "
    "video/x-raw, width={width}, height={height}, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)

def build_gst_pipeline(width=1280, height=720, fps=30, flip=0):
    """Return a GStreamer pipeline string with given parameters."""
    return GST_PIPELINE_CAMERA.format(width=width, height=height, fps=fps, flip=flip)

class Video:
    def __init__(self, frame_rate, frame_size, flip=0):
        self.cap = cv2.VideoCapture(GST_PIPELINE_CAMERA.format(width=frame_size[0], height=frame_size[1], fps=frame_rate, flip=flip), cv2.CAP_GSTREAMER)
        self.out = cv2.VideoWriter(GST_PIPELINE_STREAM, cv2.CAP_GSTREAMER, 0, frame_rate, frame_size)
        self.is_streaming = self.out.isOpened()

    def transfer(self, *frames):
        for frame in frames:
            # TODO: split frames
            pass
        self.out.write(frames[0])

    def exit(self):
        pass

    def release(self):
        self.out.release()
        self.cap.release()
