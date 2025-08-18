import cv2
import time

def run(cam):
    while True:
        frame = cam.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        #print(f"NN ran on frame shape {frame.shape}")
        time.sleep(0.2)  # simulate slow model
