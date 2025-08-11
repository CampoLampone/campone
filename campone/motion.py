import Jetson.GPIO as GPIO
import spidev

MOTOR_SPEED_BASE = 0X1A
MOTOR_POSITION_BASE = 0X1B
MOTORS_COAST = 0X30
MOTORS_BRAKE = 0X30

CS_PIN = 8

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(CS_PIN, GPIO.OUT)

def spi_transfer(spi, data):
    GPIO.output(CS_PIN, GPIO.LOW)
    padded_data = data + [0]*(8 - len(data))
    resp = spi.xfer2(padded_data)
    GPIO.output(CS_PIN, GPIO.HIGH)
    return resp

class Motion:
    LEFT, RIGHT = range(2)

    def __init__(self):
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0) # open SPI bus 0, CS 0
        self.spi.max_speed_hz = 500000   # 500 kHz
        self.spi.mode = 0

    def get_board_version(self):
        self.spi.xfer([0x00, 0x00, 0x00, 0x00])

    def set_motor_speed(self, motor, speed):
        if motor > 1 : return
        speed_16 = speed & 0xFFFF
        spi_transfer(self.spi, [MOTOR_SPEED_BASE + 0X10 * motor, (speed_16 >> 8) & 0xFF, speed_16 & 0xFF])

    def set_motor_position(self, motor, position, max_speed):
        if motor > 1 : return
        speed_16 = max_speed & 0xFFFF
        position_8 = position & 0xFF
        spi_transfer(self.spi, [MOTOR_POSITION_BASE + 0X10 * motor, position_8, (speed_16 >> 8) & 0xFF, speed_16 & 0xFF])

    def coast_motors(self):
        spi_transfer(self.spi, [MOTORS_COAST])

    def brake_motors(self):
        spi_transfer(self.spi, [MOTORS_BRAKE])

    def __del__(self):
        self.spi.close()
