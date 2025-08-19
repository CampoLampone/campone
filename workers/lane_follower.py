import time
import threading
from campone.road_processing import process, process_lines
import collections

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
Kp = 1.1
Ki = 0.0
Kd = 0.0
alpha = 0.75

# Limit
RPM_MAX = 150.0
MAX_DIFF_RPM = 100.0    # max steering contribution
INTEGRAL_LIMIT = 0.5    # correction units (after applying Ki)
DEADBAND = 0.00         # ignore small error signals
SLEW_RPM_PER_S = 400.0  # max RPM change per second for smoother commands - we'll see about this one

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

    print(time.time(), P, I, D, corr, error, L, R) # Debug line

    return L, R



class LaneFollower:
    def __init__(self, cam, freq=30):
        self.cam = cam
        self.motors = None
        self.lock = threading.Lock()
        self.freq = freq
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
            line_offset = process_lines(only_yellow, only_white)
            if line_offset == None:
                continue

            smooth_offset = self.median_filter.update(line_offset)

            output = pid_step(smooth_offset, 90)
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
