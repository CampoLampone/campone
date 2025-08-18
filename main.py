import threading
import campone

from workers.camera import CameraCapture, UDPWriter
from workers.lane_follower import LaneFollower
from workers import nn, traffic_light_detector

if __name__ == "__main__":
    cam = CameraCapture(0)
    writer = UDPWriter()
    lf = LaneFollower(cam)
    motion = campone.Motion()

    threading.Thread(target=nn.run, args=(cam,), daemon=True).start()
    threading.Thread(target=traffic_light_detector.run, args=(cam,), daemon=True).start()

    try:
        while True:
            left_mot, right_mot = lf.get_speed()
            prin(left_mot, right_mot)
            motion.set_motor_speed(motion.LEFT, -left_mot) # I dunno???
            motion.set_motor_speed(motion.RIGHT, right_mot)
            writer.show(cam.get_frame())
    except KeyboardInterrupt:
        cam.stop()
