import campone
import queue

from workers.camera import CameraCapture
from workers.lane_follower import LaneFollower
from workers.udp_socket import UdpSocket, UdpPacket

if __name__ == "__main__":
    cam = CameraCapture()
    writer = campone.UDPWriter()
    command_queue = queue.Queue(maxsize=100)
    lf = LaneFollower(cam, writer, command_queue)
    motion = campone.Motion()
    sock = UdpSocket(command_queue)

    motion.set_pid_coeffs(3, 2, 0.4)

    motors_setpoint = [0, 0]

    lf.start()

    try:
        while True:
            source, payload = command_queue.get()

            if source == 'lane_follower':
                motors = [int(x) for x in payload]
                if motors[0] != motors_setpoint[0] or motors[1] != motors_setpoint[1]:
                    motors_setpoint = motors
                    motion.set_motor_speed(motion.LEFT, -motors_setpoint[1])
                    motion.set_motor_speed(motion.RIGHT, motors_setpoint[0])

            elif source == 'udp_socket':
                packet: UdpPacket = payload
                print(f"{packet.data} from {packet.address}") # Replace this with your solution

    except KeyboardInterrupt:
        cam.stop()
        motion.brake_motors()
