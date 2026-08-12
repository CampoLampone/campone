import campone
import time

motion = campone.Motion()

test_speed = 100

# You can set PID coefficients right here
# motion.set_pid_coeffs(3, 2, 1)

# Forward
motion.set_motor_speed(motion.LEFT, -test_speed)
motion.set_motor_speed(motion.RIGHT, test_speed)
time.sleep(2)
motion.brake_motors()
time.sleep(1)
# Back
motion.set_motor_speed(motion.LEFT, test_speed)
motion.set_motor_speed(motion.RIGHT, -test_speed)
time.sleep(2)
motion.brake_motors()
