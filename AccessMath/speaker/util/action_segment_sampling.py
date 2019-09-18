
class ActionSegmentSamplingTracker:
    def __init__(self, lecture_name, segment_length):
        self.segments = []
        self.counts_per_class = {}
        self.frame_start = None
        self.lecture_name = lecture_name
        self.segment_length = segment_length

    def add_frame(self, frame_idx, label):
        if not label in self.counts_per_class:
            self.counts_per_class[label] = 1
        else:
            self.counts_per_class[label] += 1

        if self.frame_start is None:
            # first frame from next segment to track
            self.frame_start = frame_idx

        if frame_idx - self.frame_start + 1 == self.segment_length:
            # find majority class ...
            counts = sorted([(self.counts_per_class[label], label) for label in self.counts_per_class], reverse=True)
            segment_label = counts[0][1]
            # add segment ...
            self.segments.append((self.lecture_name, self.frame_start, frame_idx, segment_label))

            # reset
            self.reset()

    def reset(self):
        # ... start ...
        self.frame_start = None
        # ... and counts ...
        self.counts_per_class = {}


class ActionSegmentSampling:
    SingleLabelLeftAligned = 0
    SingleLabelCentered = 1
    MajorityLabelNoGaps = 2

    def __init__(self, sampling_mode, segment_length, tracks=1):
        self.mode = sampling_mode
        self.segment_length = segment_length
        self.tracks = tracks

    def get_speaker_file_data(self, speaker_data_filename):
        # read file ...
        with open(speaker_data_filename, "r") as input_file:
            all_text_lines = input_file.readlines()

        # frame_idx	frame_time	known	visible	label	p_0_x	p_0_y	p_1_x	p_1_y	p_2_x	p_2_y	p_3_x	p_3_y
        file_info = []
        for line in all_text_lines[1:]:
            parts = line.strip().split(",")

            if len(parts) >= 5:
                frame_idx = int(parts[0])
                known = int(parts[2]) > 0
                label = parts[4]

                file_info.append((frame_idx, known, label))

        return file_info

    def sample_from_file(self, speaker_data_filename, lecture_name):
        # get the speaker information from the original file produced by the GT annotator
        file_info = self.get_speaker_file_data(speaker_data_filename)

        # get the sample based on the current mode ...
        if self.mode in [ActionSegmentSampling.SingleLabelLeftAligned, ActionSegmentSampling.SingleLabelCentered]:
            # sampling with gap .... (same action segments only)
            # ActionSegmentSampling.SingleLabelLeftAligned = gap at the right
            # ActionSegmentSampling.SingleLabelCentered    = split gap at both sides of same action long segment
            lecture_segments = self.get_single_label_sample(file_info, lecture_name)

        elif self.mode == ActionSegmentSampling.MajorityLabelNoGaps:
            # sampling contiguously (majority label)
            # => with overlapping tracks ...
            if self.tracks == 1:
                overlaps = None
            else:
                overlaps = [int(self.segment_length * idx / self.tracks) for idx in range(1, self.tracks)]

            lecture_segments = self.get_majority_label_sample(file_info, lecture_name, overlaps)
        else:
            raise Exception("Unknown action sampling mode")

        return lecture_segments

    def get_single_label_sample(self, file_info, lecture_name):

        segments = []
        current_action = None
        current_start = None
        frame_idx = 0

        for frame_idx, known, label in file_info:
            if not known or label != current_action:
                # segment change
                # ... add any collected segment ...
                if current_action is not None:
                    segment = (lecture_name, current_start, frame_idx - 1, current_action)
                    segments.append(segment)
                # ... reset .... first of currect action (or just None)
                current_action = label if known else None
                current_start = frame_idx

        # ... add any final segment ...
        if current_action is not None:
            segment = (lecture_name, current_start, frame_idx, current_action)
            segments.append(segment)

        # ... sample from all frames ...
        final_segments = []

        for lecture_name, segment_start, segment_end, action in segments:
            # ... if there are enough frames in the segment ...
            segment_frames = segment_end - segment_start + 1
            total_sub_segments = int(segment_frames / self.segment_length)
            if total_sub_segments > 0:
                # ... sample based on sampling strategy ...
                if self.mode == ActionSegmentSampling.SingleLabelLeftAligned:
                    # use segments continuously ... from beginning until no more segments can be sampled ...
                    base_start = segment_start
                    base_skip = 0
                elif self.mode == ActionSegmentSampling.SingleLabelCentered:
                    # use segments continuously ... by balancing the gap at the sides of the segment ...
                    segment_gap = segment_frames - total_sub_segments * self.segment_length
                    base_start = segment_start + int(segment_gap / 2)
                    base_skip = 0
                else:
                    raise Exception("Unknown action segment sampling mode ... ")

                for idx in range(total_sub_segments):
                    current_start = base_start + (idx * (self.segment_length + base_skip))
                    current_end = current_start + self.segment_length - 1
                    # add segment ...
                    final_segments.append((lecture_name, current_start, current_end, action))

        # for lecture_name, segment_start, segment_end, action in final_segments:
        #     print((lecture_name, segment_start, segment_end, action))

        return final_segments

    def get_majority_label_sample(self, file_info, lecture_name, overlapping=None):
        trackers = [(0, ActionSegmentSamplingTracker(lecture_name, self.segment_length))]

        if overlapping is not None:
            trackers += [(offset, ActionSegmentSamplingTracker(lecture_name, self.segment_length)) for offset in overlapping]

        for frame_idx, known, label in file_info:
            if known:
                for offset, tracker in trackers:
                    if offset <= frame_idx:
                        tracker.add_frame(frame_idx, label)
            else:
                for offset, tracker in trackers:
                    tracker.reset()

        final_segments = []
        for offset, tracker in trackers:
            final_segments += tracker.segments

        return final_segments
