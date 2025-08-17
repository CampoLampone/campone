__version__ = "1.0.0"

def is_jetson_nano():
    try:
        with open("/sys/firmware/devicetree/base/model", "r") as f:
            model = f.read().strip('\x00').strip()
            return "Jetson Nano" in model
    except FileNotFoundError:
        return False


_on_jetson = is_jetson_nano()

print("Running on Jetson Nano:", _on_jetson)

if _on_jetson:
    from .motion import Motion
    from .video import Video

from .road_processing import process_lines, is_intersection, process

__all__ = ["Motion", "Video", "process_lines", "is_intersection", "process"]
