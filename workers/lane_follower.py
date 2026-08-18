import collections
import threading
import time

import numpy as np

from campone.road_processing import (
    get_clear_lines,
    get_line_centroids,
    process,
    process_lines,
    visualize_line_offset,
    visualize_motor_push,
    visualize_point_array,
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
Kp = 0.55
Ki = 0.3
Kd = 0.05
alpha = 0.05

# Limit
RPM_MAX = 150.0
MAX_DIFF_RPM = 100.0 # max steering contribution
INTEGRAL_LIMIT = 1 # correction units (after applying Ki)
DEADBAND = 0.02 # ignore small error signals
SLEW_RPM_PER_S = 400.0 # max RPM change per second for smoother commands - we'll see about this one

def clamp(x, lo, hi): return max(lo, min(hi, x))

def slew(prev, target, max_step):
    if target > prev + max_step: return prev + max_step
    if target < prev - max_step: return prev - max_step
    return target

class LaneFollowerWorker:
    def __init__(self, cam, command_queue, stream_worker=None, base_speed=50):
        self.cam = cam
        self.command_queue = command_queue
        self.stream_worker = stream_worker
        self.base_speed = base_speed

        self.running = False

        self.reset_state()

    def reset_state(self):
        self._integral = 0.0
        self._d = 0.0
        self._last_e = 0.0
        self._last_t = time.time()
        self._last_L = 0.0
        self._last_R = 0.0
        self.median_filter = MedianFilter(win_size=3)

    def start(self, reset = True):
        if reset: self.reset_state()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.running = True
        self.thread.start()

    def pid_step(self, error, base_rpm):
        """
        error: normalized deviation [-1, 1]
        base_rpm: forward speed (-150..150)
        returns: (left_rpm, right_rpm)
        """

        t = time.time()
        dt = clamp(t - self._last_t, 1e-3, 5e-2)

        # Apply deadband
        filtered_error = error
        if abs(filtered_error) < DEADBAND:
            filtered_error = 0.0

        # PID terms
        P = Kp * filtered_error

        self._integral += error * dt
        if Ki != 0:
            self._integral = clamp(self._integral, -INTEGRAL_LIMIT / Ki, INTEGRAL_LIMIT / Ki)
        I = Ki * self._integral

        de = (error - self._last_e) / dt
        self._d = alpha * self._d + (1 - alpha) * de
        D = Kd * self._d

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
        self._last_e, self._last_t = error, t
        self._last_L, self._last_R = L, R

        # print(time.time(), P, I, D, corr, error, L, R) # Debug line

        return L, R

    def run(self):
        self._last_t = time.time() # Reset so D doesn't explode
        last_frame_id = None
        while self.running:
            frame, last_frame_id = self.cam.wait_for_frame(last_frame_id)
            if frame is None or not self.running:
                continue

            only_yellow, only_white = process(frame)
            frame_shape = only_yellow.shape
            only_yellow, only_white = get_clear_lines(only_yellow, only_white)

            debug_out1 = np.zeros(np.append(only_yellow.shape, 3))
            debug_out1[:, :, 0] = only_yellow
            debug_out1[:, :, 1] = only_white

            left_points, right_points = get_line_centroids(only_yellow, only_white)
            if left_points is None and right_points is None:
                if self.stream_worker is not None:
                    self.stream_worker.set_frame("lane_main", frame.copy())
                continue

            visualize_point_array(debug_out1, left_points)
            visualize_point_array(debug_out1, right_points)

            line_offset = process_lines(frame_shape, left_points, right_points)

            visualize_line_offset(frame, line_offset)

            if line_offset == None:
                if self.stream_worker is not None:
                    self.stream_worker.set_frame("lane_main", frame.copy())
                continue


            # smooth_offset = self.median_filter.update(line_offset)

            output = self.pid_step(line_offset, self.base_speed)
            # output = self.pid_step(smooth_offset, self.base_speed)

            # Visualizing motor output
            right = output[1]
            left = output[0]

            visualize_motor_push(frame, left, right)

            if self.stream_worker is not None:
                self.stream_worker.set_frame("lane_main", frame.copy())
                self.stream_worker.set_frame("lane_debug", debug_out1)

            self.command_queue.put(('motor_command', output))

    def stop(self):
        if not self.running:
            return

        if self.stream_worker is not None:
            self.stream_worker.clear_frame("lane_main")
            self.stream_worker.clear_frame("lane_debug")
        self.running = False
        self.thread.join()
        del(self.thread) # Threads can only be started once
        self.command_queue.put(('motor_command', "brake")) # Stop the motors
