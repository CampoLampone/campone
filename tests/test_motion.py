import campone
import time

motion = campone.Motion()
motion.set_motor_speed(motion.LEFT, 500)
motion.set_motor_speed(motion.RIGHT, 500)
print("500")
time.sleep(3)
motion.set_motor_speed(motion.LEFT, 200)
motion.set_motor_speed(motion.RIGHT, 200)
print("200")
time.sleep(3)
motion.set_motor_speed(motion.LEFT, 10)
motion.set_motor_speed(motion.RIGHT, 10)
print("10")
time.sleep(3)
motion.brake_motors()
