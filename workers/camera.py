import cv2
import numpy as np
import threading
import os
import time
import campone

GST_PIPELINE_CAMERA = (
    "nvarguscamerasrc wbmode=6 sensor-mode=3 ! "
    "video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1, format=NV12 ! "
    "nvvidconv flip-method={flip} ! "
    "video/x-raw, width={width}, height={height}, format=BGRx ! appsink drop=1 max-buffers=1 sync=0"
)


class FakeCapture:
    def __init__(self, filename):
        moduledir = os.path.dirname(__file__)
        filepath = os.path.join(moduledir, "test_images", filename)
        self.img = cv2.imread(filepath)

    def read(self):
        time.sleep(1/30.0)  # Simulate 30 FPS to prevent 100% CPU usage
        return True, self.img

    def release(self):
        pass


class CameraCaptureWorker:
    def __init__(self):
        self.frame_rate = 30
        self.flip = 0
        self.frame_size = (1280, 720)

        if not campone._on_jetson:
            self.cap = FakeCapture("testing.png")
        else:
            self.cap = cv2.VideoCapture(GST_PIPELINE_CAMERA.format(width=self.frame_size[0], height=self.frame_size[1], fps=self.frame_rate, flip=self.flip), cv2.CAP_GSTREAMER)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.frame = None
        self.frame_id = 0
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
            with self.condition:
                self.frame = undistorted_frame
                self.frame_id += 1
                self.condition.notify_all()

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def wait_for_frame(self, last_frame_id=None, timeout=None):
        with self.condition:
            if last_frame_id is not None:
                self.condition.wait_for(lambda: self.frame_id != last_frame_id or not self.running, timeout=timeout)
            else:
                self.condition.wait(timeout=timeout)
            return (None if self.frame is None else self.frame.copy()), self.frame_id

    def stop(self):
        self.running = False
        with self.condition:
            self.condition.notify_all()  # Wake up any waiting threads so they can exit
        self.thread.join()
        self.cap.release()
