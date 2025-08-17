import cv2
import math
import numpy as np
import socket

GST_PIPELINE_STREAM = (
    "appsrc caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 "
    "is-live=true block=true format=TIME do-timestamp=true ! queue ! videoconvert ! queue !"
    "nvvidconv ! queue ! nvv4l2h264enc insert-sps-pps=true iframeinterval=15 idrinterval=30 bitrate=800000 ! queue ! "
    "h264parse ! mpegtsmux ! "
    "udpsink host={host} port=5000 sync=false async=false"
)

GST_PIPELINE_CAMERA = (
    "nvarguscamerasrc sensor-mode=3 ! "
    "video/x-raw(memory:NVMM), width={width}, height={height}, exposuretimerange=(string)2000000 2000000, framerate={fps}/1, format=NV12 ! "
    "nvvidconv flip-method={flip} ! "
    "video/x-raw, width={width}, height={height}, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)

MAX_FRAME_SIZE = (720, 1280, 3)

def build_gst_pipeline(width=1280, height=720, fps=30, flip=0):
    """Return a GStreamer pipeline string with given parameters."""
    return GST_PIPELINE_CAMERA.format(width=width, height=height, fps=fps, flip=flip)

def get_subscriber_hostname():
    return "team" + socket.gethostname()[-1] + ".lan"

class Video:
    def __init__(self, frame_rate, frame_size, flip=0):
        self.cap = cv2.VideoCapture(GST_PIPELINE_CAMERA.format(width=frame_size[0], height=frame_size[1], fps=frame_rate, flip=flip), cv2.CAP_GSTREAMER)
        self.out = cv2.VideoWriter(GST_PIPELINE_STREAM.format(width=frame_size[0], height=frame_size[1], fps=frame_rate, host=get_subscriber_hostname()), cv2.CAP_GSTREAMER, 0, frame_rate, frame_size, True)

        self.is_streaming = self.out.isOpened()
        self.is_recording = self.cap.isOpened()

        if not self.is_streaming:
            print("Error: Could not open video writer")
            exit()
        if not self.is_recording:
            print("Error: Could not open camera recorder")
            exit()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret: print("Frame not retrieved!")
        return frame

    def show(self, *frames):
        n = len(frames)
        r = math.ceil(math.sqrt(n))

        tile_h = MAX_FRAME_SIZE[0] // r
        tile_w = MAX_FRAME_SIZE[1] // r

        out_frame = np.zeros(MAX_FRAME_SIZE, dtype=np.uint8)

        i_imgs = 0
        for i_y in range(r):
            for i_x in range(r):
                if i_imgs >= n:
                    break
                downscaled = cv2.resize(frames[i_imgs], (tile_w, tile_h), interpolation=cv2.INTER_AREA)
                if downscaled.ndim == 2:
                    downscaled = cv2.cvtColor(downscaled, cv2.COLOR_GRAY2BGR)

                y1, y2 = i_y * tile_h, (i_y + 1) * tile_h
                x1, x2 = i_x * tile_w, (i_x + 1) * tile_w
                out_frame[y1:y2, x1:x2] = downscaled

                i_imgs += 1

        self.out.write(out_frame)

    def exit(self):
        pass

    def release(self):
        self.out.release()
        self.cap.release()
