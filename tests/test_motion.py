import campone
import time

rt_board = campone.RTBoard()

test_speed = 100

# You can set PID coefficients right here
# rt_board.set_pid_coeffs(3, 2, 1)

# Forward
rt_board.set_motor_speed(rt_board.Motor.LEFT, -test_speed)
rt_board.set_motor_speed(rt_board.Motor.RIGHT, test_speed)
time.sleep(2)
rt_board.brake_motors()
time.sleep(1)
# Back
rt_board.set_motor_speed(rt_board.Motor.LEFT, test_speed)
rt_board.set_motor_speed(rt_board.Motor.RIGHT, -test_speed)
time.sleep(2)
rt_board.brake_motors()
