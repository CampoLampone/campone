import cv2
from campone import stream
import time
from workers.camera import CameraCapture

cam = CameraCapture()
writer = stream.UDPWriter()

while True:
    frame = cam.get_frame()
    if frame is not None:
        writer.show(frame)
