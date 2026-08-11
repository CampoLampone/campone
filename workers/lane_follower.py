import time
import threading
from campone.road_processing import process, process_lines
import collections
import numpy as np

from campone.road_processing import process, get_clear_lines, process_lines, get_line_centroids
from campone.road_processing import (
    visualize_point_array,
    visualize_line_offset,
    visualize_motor_push
)

class MedianFilter:
    def __init__(self, win_size):
        self.win_size = win_size
        self.window = collections.deque(maxlen=self.win_size)

    def update(self, new_value):
        self.window.append(new_value)
        if len(self.window) < self.win_size:
            return new_value  # Not enough data yet
        sorted_window = sorted(self.window)
        return sorted_window[self.win_size // 2]

# PID coefficients
Kp = 0.4
Ki = 0.01
Kd = 0.00
alpha = 0.2

# Limit
RPM_MAX = 150.0
MAX_DIFF_RPM = 100.0 # max steering contribution
INTEGRAL_LIMIT = 0.5 # correction units (after applying Ki)
DEADBAND = 0.02 # ignore small error signals
SLEW_RPM_PER_S = 400.0 # max RPM change per second for smoother commands - we'll see about this one

_integral = 0.0
_d = 0.0
_last_e = 0.0
_last_t = time.time()
_last_L = 0.0
_last_R = 0.0

def clamp(x, lo, hi): return max(lo, min(hi, x))

def slew(prev, target, max_step):
    if target > prev + max_step: return prev + max_step
    if target < prev - max_step: return prev - max_step
    return target

def pid_step(error, base_rpm):
    """
    error: normalized deviation [-1, 1]
    base_rpm: forward speed (-150..150)
    returns: (left_rpm, right_rpm)
    """
    global _integral, _d, _last_e, _last_t, _last_L, _last_R

    t = time.time()
    dt = max(1e-3, t - _last_t)

    # Apply deadband
    if abs(error) < DEADBAND:
        error = 0.0

    # PID terms
    P = Kp * error

    _integral += error * dt
    I = Ki * _integral
    I = clamp(I, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)

    de = (error - _last_e) / dt
    _d = alpha * _d + (1 - alpha) * de
    D = Kd * _d

    corr = P + I + D
    corr = clamp(corr, -1.0, 1.0)

    # Map to differential RPM
    L = base_rpm + corr * MAX_DIFF_RPM
    R = base_rpm - corr * MAX_DIFF_RPM

    # Clamp to motor RPM capability
    L = clamp(L, -RPM_MAX, RPM_MAX)
    R = clamp(R, -RPM_MAX, RPM_MAX)

    # Slew-rate limiting
    # max_step = SLEW_RPM_PER_S * dt
    # L = slew(_last_L, L, max_step)
    # R = slew(_last_R, R, max_step)

    # Save state
    _last_e, _last_t = error, t
    _last_L, _last_R = L, R

    # print(time.time(), P, I, D, corr, error, L, R) # Debug line

    return L, R



class LaneFollower:
    def __init__(self, cam, writer, base_speed=50, freq=40):
        self.cam = cam
        self.writer = writer
        self.motors = None
        self.lock = threading.Lock()
        self.freq = freq
        self.base_speed = base_speed
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

        self.median_filter = MedianFilter(win_size=3)

    def run(self):
        while self.running:
            start_time = time.time()
            frame = self.cam.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            only_yellow, only_white = process(frame)
            frame_shape = only_yellow.shape
            only_yellow, only_white = get_clear_lines(only_yellow, only_white)

            debug_out1 = np.zeros(np.append(only_yellow.shape, 3))
            debug_out1[:, :, 0] = only_yellow
            debug_out1[:, :, 1] = only_white

            left_points, right_points = get_line_centroids(only_yellow, only_white)
            if left_points is None and right_points is None:
                continue

            visualize_point_array(debug_out1, left_points)
            visualize_point_array(debug_out1, right_points)

            line_offset = process_lines(frame_shape, left_points, right_points)

            visualize_line_offset(frame, line_offset)

            if line_offset == None:
                continue

            smooth_offset = self.median_filter.update(line_offset)

            output = pid_step(smooth_offset, self.base_speed)

            # Visualizing motor output
            right = output[0]
            left = output[1]

            print(left, right)

            visualize_motor_push(frame, left, right)

            self.writer.show(frame, debug_out1)

            with self.lock:
                self.motors = output
            time_delta = time.time() - start_time
            if (time_delta < (1 / self.freq)):
                time.sleep(time_delta)

    def get_speed(self):
        with self.lock:
            return self.motors

    def stop(self):
        self.running = False
        self.thread.join()
