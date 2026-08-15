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
    "h264parse config-interval=1 ! mpegtsmux alignment=7 ! "
    "srtsink uri=srt://:5000 mode=listener wait-for-connection=false sync=false async=false latency=30"
)

MAX_FRAME_SIZE = (720, 1280, 3)


class VideoStreamer:
    def __init__(self, frame_size: tuple[int, int] = (1280, 720), frame_rate: int = 30):
        self.frame_rate = frame_rate
        self.frame_size = frame_size
        self.out = cv2.VideoWriter(GST_PIPELINE_STREAM.format(width=self.frame_size[0], height=self.frame_size[1], fps=self.frame_rate), cv2.CAP_GSTREAMER, 0, self.frame_rate, self.frame_size, True)
        self.out_frame = np.zeros(MAX_FRAME_SIZE, dtype=np.uint8)

    def show(self, *frames):
        n = len(frames)
        if n != 1:
            r = math.ceil(math.sqrt(n))

            tile_h = MAX_FRAME_SIZE[0] // r
            tile_w = MAX_FRAME_SIZE[1] // r

            self.out_frame[:] = 0

            for i, frame in enumerate(frames):
                if frame is None:
                    continue

                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]

                downscaled = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)

                if downscaled.ndim == 2:
                    downscaled = cv2.cvtColor(downscaled, cv2.COLOR_GRAY2BGR)

                i_y, i_x = divmod(i, r)
                y1, y2 = i_y * tile_h, (i_y + 1) * tile_h
                x1, x2 = i_x * tile_w, (i_x + 1) * tile_w
                self.out_frame[y1:y2, x1:x2] = downscaled

            self.out.write(self.out_frame)
        else:
            one_frame = frames[0]
            if one_frame.ndim == 3 and one_frame.shape[2] == 4:
                one_frame = one_frame[:, :, :3]
            if one_frame.shape[:2][::-1] != self.frame_size:
                one_frame = cv2.resize(one_frame, self.frame_size, interpolation=cv2.INTER_LINEAR)
            self.out.write(one_frame)
