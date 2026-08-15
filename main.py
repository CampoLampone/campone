import campone
import queue
import threading

from workers.camera import CameraCaptureWorker
from workers.lane_follower import LaneFollowerWorker
from workers.udp_socket import UdpSocketWorker, UdpPacket
from workers.stream_worker import StreamWorker

if __name__ == "__main__":
    camera_worker = CameraCaptureWorker()
    command_queue = queue.Queue(maxsize=100)

    stream_worker = StreamWorker()
    stream_worker.start()

    lf = LaneFollowerWorker(camera_worker, command_queue, stream_worker)
    rt_board = campone.RTBoard()
    sock_worker = UdpSocketWorker(command_queue)

    # You can tune the Real-Time board's PID right here in runtime
    # rt_board.set_pid_coeffs(3, 2, 0.4)

    motors_setpoint = [0, 0]

    lf.start()

    try:
        while True:
            source, payload = command_queue.get()

            if source == 'motor_command':
                motors = [int(x) for x in payload]
                if motors[0] != motors_setpoint[0] or motors[1] != motors_setpoint[1]:
                    motors_setpoint = motors
                    rt_board.set_motor_speed(rt_board.Motor.LEFT, -motors_setpoint[1])
                    rt_board.set_motor_speed(rt_board.Motor.RIGHT, motors_setpoint[0])

            elif source == 'udp_socket':
                packet: UdpPacket = payload
                print(f"{packet.data} from {packet.address}") # Replace this with your solution

    except KeyboardInterrupt:
        camera_worker.stop()
        rt_board.brake_motors()
        stream_worker.stop()
