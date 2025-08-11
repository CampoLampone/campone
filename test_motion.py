import campone
import time

motion = campone.Motion()
motion.set_motor_speed(motion.LEFT, 500)
motion.set_motor_speed(motion.RIGHT, 500)
time.sleep(1)
motion.brake_motors()
