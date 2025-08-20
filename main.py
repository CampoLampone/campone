import threading
import campone

from workers.camera import CameraCapture
from workers.lane_follower import LaneFollower
from campone.road_processing import process, process_lines, is_intersection
# from workers import nn, traffic_light_detector # - disabled for now

if __name__ == "__main__":
    cam = CameraCapture(0)
    writer = campone.UDPWriter()
    lf = LaneFollower(cam)
    motion = campone.Motion()

    # disabled for now
    # threading.Thread(target=nn.run, args=(cam,), daemon=True).start()
    # threading.Thread(target=traffic_light_detector.run, args=(cam,), daemon=True).start()

    motors_setpoint = [0, 0]

    try:
        while True:
            motors = lf.get_speed()
            if motors is None: continue
            motors = [int(x) for x in motors]

            # Only update motors if the current speed is different from the last setpoint
            if motors[0] != motors_setpoint[0] or motors[1] != motors_setpoint[1]:
                motors_setpoint = motors  # Update the setpoint to the new speed
                motion.set_motor_speed(motion.LEFT, -motors_setpoint[1])
                motion.set_motor_speed(motion.RIGHT, motors_setpoint[0])

            img = cam.get_frame()
            only_yellow, only_white = process(img)

            if is_intersection(only_yellow):
                print("balls")

            writer.show(img[:, :, :3], only_yellow, only_white)
    except KeyboardInterrupt:
        cam.stop()
        motion.brake_motors()
