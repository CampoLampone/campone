import threading
import time
from workers.camera import CameraCapture
from workers.lane_follower import LaneFollower
from workers import nn, traffic_light_detector

if __name__ == "__main__":
    cam = CameraCapture(0)

    lf = LaneFollower(cam)
    threading.Thread(target=nn.run, args=(cam,), daemon=True).start()
    threading.Thread(target=traffic_light_detector.run, args=(cam,), daemon=True).start()

    try:
        while True:
            time.sleep(1)
            print(lf.get_speed())
    except KeyboardInterrupt:
        cam.stop()
