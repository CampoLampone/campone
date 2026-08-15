import time

import cv2

from campone.stream import VideoStreamer
from workers.camera import CameraCaptureWorker

cam = CameraCaptureWorker()
writer = VideoStreamer()

while True:
    frame = cam.get_frame()
    if frame is not None:
        writer.show(frame)
