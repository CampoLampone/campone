import cv2
from campone import stream
import time
import atexit

GST_PIPELINE_CAMERA = (
    "nvarguscamerasrc wbmode=6 sensor-mode=3 ! "
    "video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1, format=NV12 ! "
    "nvvidconv flip-method={flip} ! "
    "video/x-raw, width={width}, height={height}, format=BGRx ! appsink drop=1 max-buffers=1 sync=0"
)

frame_rate = 30
flip = 0
frame_size = (1280, 720)
cap = cv2.VideoCapture(GST_PIPELINE_CAMERA.format(width=frame_size[0], height=frame_size[1], fps=frame_rate, flip=flip), cv2.CAP_GSTREAMER)

writer = stream.UDPWriter()

atexit.register(cap.release)

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        continue
    writer.show(frame[:, :, :3])

    time_delta = time.time() - start_time
    if (time_delta < (1 / frame_rate)):
        time.sleep(time_delta)
