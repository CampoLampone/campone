import threading
import campone
import time

from workers.camera import CameraCapture
from workers.lane_follower import LaneFollower
from campone.road_processing import process, is_intersection
# from workers import nn, traffic_light_detector # - disabled for now

if __name__ == "__main__":
    cam = CameraCapture()
    writer = campone.UDPWriter()
    lf = LaneFollower(cam, writer)
    motion = campone.Motion()

    # disabled for now
    # threading.Thread(target=nn.run, args=(cam,), daemon=True).start()
    # threading.Thread(target=traffic_light_detector.run, args=(cam,), daemon=True).start()

    motors_setpoint = [0, 0]

    try:
        while True:
            motors = lf.get_speed()
            if motors is None: 
                time.sleep(0.01)
                continue
            motors = [int(x) for x in motors]

            # Only update motors if the current speed is different from the last setpoint
            if motors[0] != motors_setpoint[0] or motors[1] != motors_setpoint[1]:
                motors_setpoint = motors  # Update the setpoint to the new speed
                # Note, notation is: [right, -left]
                motion.set_motor_speed(motion.LEFT, -motors_setpoint[1])
                motion.set_motor_speed(motion.RIGHT, motors_setpoint[0])

    except KeyboardInterrupt:
        cam.stop()
        motion.brake_motors()
