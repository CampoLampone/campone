import campone
import time

motion = campone.Motion()
motion.release_estop()
motion.set_motor_speed(motion.LEFT, -150)
motion.set_motor_speed(motion.RIGHT, 150)
time.sleep(3)
motion.set_motor_speed(motion.LEFT, 150)
motion.set_motor_speed(motion.RIGHT, -150)
time.sleep(3)
motion.brake_motors()
