import socket
import threading
import queue
from dataclasses import dataclass

@dataclass
class UdpPacket:
    data: bytes = b''
    address: str = "0.0.0.0"

class UdpSocketWorker:
    def __init__(self, command_queue):
        self.host = '0.0.0.0'
        self.port = 6969
        self.command_queue = command_queue
        self.max_message_size = 16

        self.thread = threading.Thread(target=self.run, daemon=True)
        self.running = True
        self.thread.start()

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        sock.settimeout(1.0) # so we can check self.running at least once every second
        while self.running:
            try:
                data, address = sock.recvfrom(self.max_message_size)
                try:
                    self.command_queue.put_nowait(('udp_socket', UdpPacket(data=data, address=address[0])))
                except queue.Full:
                    pass # probably getting spammed by some very funny participant
            except TimeoutError:
                continue

        sock.close()

    def __del__(self):
        self.running = False
        self.thread.join()
