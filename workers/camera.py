import cv2
import numpy as np
import threading
import os
import campone

# Define the distortion coefficients
k1 = -0.012  # Radial distortion coefficient
k2 = 0.00012  # Radial distortion coefficient
p1 = -0.0013  # Tangential distortion coefficient
p2 = 0.0015  # Tangential distortion coefficient

# Define the parameters for manual correction
fov = 160  # Field of view (in degrees)

new_size = (1280, 720)

# Calculate the focal length based on the field of view
focal_length = new_size[0] / (2 * np.tan(np.radians(fov) / 2))

# Generate a simple perspective transformation matrix
K = np.array([[focal_length, 0, new_size[0] / 2], [0, focal_length, new_size[1] / 2], [0, 0, 1]])
dist_coeffs = np.array([k1, k2, p1, p2])


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
        if not campone._on_jetson:
            self.cap = FakeCapture("testing.png")
        else:
            self.cap = cv2.VideoCapture(f'nvarguscamerasrc sensor-mode=3 ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)29/1 ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx ! videoconvert ! appsink')
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            # Resize to predefied size
            frame = cv2.resize(frame, new_size)
            # Undistort the image using the specified coefficients
            undistorted_frame = cv2.undistort(frame, K, dist_coeffs)
            with self.lock:
                self.frame = undistorted_frame.copy()

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
