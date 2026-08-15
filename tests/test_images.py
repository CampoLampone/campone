import sys
import cv2
import curses

sys.path.append('../workers')
from workers.camera import CameraCaptureWorker
from campone.stream import VideoStreamer


def capture_loop(stdscr):
    cam = CameraCaptureWorker()
    writer = VideoStreamer()

    curses.curs_set(0)       # hide cursor
    stdscr.nodelay(True)     # non-blocking input
    stdscr.clear()
    stdscr.addstr(0, 0, "Press SPACE to capture an image, Q to exit.")
    stdscr.refresh()

    img_counter = 0

    while True:
        frame = cam.get_frame()
        if frame is None:
            stdscr.addstr(2, 0, "Failed to grab frame.\n")
            stdscr.refresh()
            continue

        key = stdscr.getch()

        if key == ord("q") or key == ord("Q"):  # Close
            break
        elif key == ord(" "):  # SPACE
            img_counter += 1
            filename = f"capture_{img_counter}.jpg"
            cv2.imwrite(filename, frame)
            stdscr.addstr(2, 0, f"Captured {filename} and sent to stream.   ")
            stdscr.refresh()

        writer.show(frame)

    cam.stop()


def main():
    curses.wrapper(capture_loop)


if __name__ == "__main__":
    main()
