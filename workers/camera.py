import cv2
import numpy as np
import threading
import os
import socket
import math
import campone

# Pipelines
GST_PIPELINE_STREAM = (
    "appsrc caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 "
    "is-live=true block=true format=TIME do-timestamp=true ! queue ! videoconvert ! queue !"
    "nvvidconv ! queue ! nvv4l2h264enc insert-sps-pps=true iframeinterval=15 idrinterval=30 bitrate=800000 ! queue ! "
    "h264parse ! mpegtsmux ! "
    "udpsink host={host} port=5000 sync=false async=false"
)

GST_PIPELINE_CAMERA = (
    "nvarguscamerasrc wbmode=6 sensor-mode=3 ! "
    "video/x-raw(memory:NVMM), width={width}, height={height}, exposuretimerange=(string)2000000 2000000, framerate={fps}/1, format=NV12 ! "
    "nvvidconv flip-method={flip} ! "
    "video/x-raw, width={width}, height={height}, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)

MAX_FRAME_SIZE = (720, 1280, 3)

class FakeCapture:
    def __init__(self, filename):
        moduledir = os.path.dirname(__file__)
        filepath = os.path.join(moduledir, "test_images", filename)
        self.img = cv2.imread(filepath)

    def read(self):
        return True, self.img

    def release(self):
        pass


class CameraCapture:
    def __init__(self, src=0):
        self.frame_rate = 30
        self.flip = 0
        self.frame_size = (1280, 720)

        if not campone._on_jetson:
            self.cap = FakeCapture("testing.png")
        else:
            self.cap = cv2.VideoCapture(GST_PIPELINE_CAMERA.format(width=self.frame_size[0], height=self.frame_size[1], fps=self.frame_rate, flip=self.flip), cv2.CAP_GSTREAMER)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True

        fov = 160
        focal_length = self.frame_size[0] / (2 * np.tan(np.radians(fov) / 2))
        K = np.array([[focal_length, 0, self.frame_size[0] / 2],
                    [0, focal_length, self.frame_size[1] / 2],
                    [0, 0, 1]])
        init_k1 = -0.012
        init_k2 = 0.00012
        init_p1 = -0.0013
        init_p2 = 0.0015

        dist_coefs = np.array([init_k1, init_k2, init_p1, init_p2])
        self.map1, self.map2 = cv2.initUndistortRectifyMap(K, dist_coefs, None, K, self.frame_size, cv2.CV_16SC2)

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            # Resize to predefied size
            # frame = cv2.resize(frame, new_size) # type: ignore
            # Undistort the image using the specified coefficients
            undistorted_frame = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
            with self.lock:
                self.frame = undistorted_frame.copy()

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()


class UDPWriter:
    def __init__(self):
        self.frame_rate = 30
        self.frame_size = (1280, 720)
        self.out = cv2.VideoWriter(GST_PIPELINE_STREAM.format(width=self.frame_size[0], height=self.frame_size[1], fps=self.frame_rate, host=self.get_subscriber_hostname()), cv2.CAP_GSTREAMER, 0, self.frame_rate, self.frame_size, True)

    def get_subscriber_hostname(self):
        return "danusasus" + ".lan"    

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
