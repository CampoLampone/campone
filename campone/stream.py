import cv2
import math
import numpy as np
import socket

GST_PIPELINE_STREAM = (
    "appsrc caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 "
    "is-live=true block=true format=TIME do-timestamp=true ! "
    "videoconvert ! nvvidconv ! "
    "nvv4l2h264enc insert-sps-pps=true iframeinterval=15 idrinterval=15 "
    "control-rate=1 preset-level=1 bitrate=2000000 ! "
    "h264parse config-interval=1 ! rtph264pay pt=96 ! "
    "udpsink host={host} port=5000 sync=false async=false"
)

MAX_FRAME_SIZE = (720, 1280, 3)


class UDPWriter:
    def __init__(self):
        self.frame_rate = 30
        self.frame_size = (1280, 720)
        self.out = cv2.VideoWriter(GST_PIPELINE_STREAM.format(width=self.frame_size[0], height=self.frame_size[1], fps=self.frame_rate, host=self.get_subscriber_hostname()), cv2.CAP_GSTREAMER, 0, self.frame_rate, self.frame_size, True)
        self.out_frame = np.zeros(MAX_FRAME_SIZE, dtype=np.uint8)

    def get_subscriber_hostname(self):
        return "team" + socket.gethostname()[-1] + ".lan"

    def show(self, *frames):
        n = len(frames)
        if n != 1:
            r = math.ceil(math.sqrt(n))

            tile_h = MAX_FRAME_SIZE[0] // r
            tile_w = MAX_FRAME_SIZE[1] // r

            self.out_frame[:] = 0

            i_imgs = 0
            for i_y in range(r):
                for i_x in range(r):
                    if i_imgs >= n:
                        break
                    frame = frames[i_imgs]
                    if frame is None:
                        continue
                    if len(frame.shape) == 3 and frame.shape[2] == 4:
                        frame = frame[:, :, :3]
                    downscaled = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
                    if downscaled.ndim == 2:
                        downscaled = cv2.cvtColor(downscaled, cv2.COLOR_GRAY2BGR)

                    y1, y2 = i_y * tile_h, (i_y + 1) * tile_h
                    x1, x2 = i_x * tile_w, (i_x + 1) * tile_w
                    self.out_frame[y1:y2, x1:x2] = downscaled

                    i_imgs += 1

            self.out.write(self.out_frame)
        else:
            one_frame = frames[0]
            if one_frame.shape[:2][::-1] != self.frame_size:
                one_frame = cv2.resize(one_frame, self.frame_size, interpolation=cv2.INTER_AREA)
            self.out.write(one_frame)
