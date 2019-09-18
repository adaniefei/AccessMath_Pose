
import cv2

class SimpleFrameSampler:
    def __init__(self):
        self.width = None
        self.height = None

        self.frame_times = None
        self.frame_indices = None
        self.compressed_frames = None

    def initialize(self, width, height):
        self.width = width
        self.height = height

        self.frame_times = []
        self.frame_indices = []
        self.compressed_frames = []

    def handleFrame(self, frame, last_frame, v_index, abs_time, rel_time, abs_frame_idx):
        flag, raw_data = cv2.imencode(".png", frame)
        self.compressed_frames.append(raw_data)
        self.frame_indices.append(abs_frame_idx)
        self.frame_times.append(abs_time)

    def getWorkName(self):
        return "Simple Frame Sampler"

    def finalize(self):
        pass

