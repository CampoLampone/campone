import campone
import time

rt_board = campone.RTBoard()

brightness = 255

rt_board.set_led_rgb(brightness,0,0)
time.sleep(0.5)
rt_board.set_led_rgb(brightness,brightness,0)
time.sleep(0.5)
rt_board.set_led_rgb(0,brightness,0)
time.sleep(0.5)
rt_board.set_led_rgb(0,brightness,brightness)
time.sleep(0.5)
rt_board.set_led_rgb(0,0,brightness)
time.sleep(0.5)
rt_board.set_led_rgb(brightness,0,brightness)
time.sleep(0.5)
rt_board.set_led_rgb(0,0,0)
