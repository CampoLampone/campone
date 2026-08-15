import threading
import time
import campone

class StreamWorker:
    def __init__(self, fps=30):
        self.streamer = campone.VideoStreamer(frame_rate=fps)
        self.latest_frames = {}
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def set_frame(self, source_id: str, frame):
        """Update the latest frame for a given source."""
        with self.lock:
            self.latest_frames[source_id] = frame

    def clear_frame(self, source_id: str):
        """Clear frame for a given source."""
        with self.lock:
            if source_id in self.latest_frames:
                del self.latest_frames[source_id]

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while self.running:
            with self.lock:
                frames = list(self.latest_frames.values())

            if frames:
                self.streamer.show(*frames)
            else:
                # Sleep briefly to avoid spinning CPU if no frames are available yet
                time.sleep(0.01)

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
