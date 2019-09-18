
import cv2
import numpy as np

from scipy.interpolate import interp1d
import scipy.ndimage.measurements as sci_mes

from AccessMath.preprocessing.content.MLBinarizer import MLBinarizer

from AccessMath.preprocessing.content.keyframe_exporter import KeyframeExporter


class SummaryGenerator:
    def __init__(self, config, verbose=True):
        self.ml_binarizer = MLBinarizer.FromConfig(config)

        # Percentage of CC pixels that should be present on a binary mask to remove them from the binary image
        # 0.75 based on experiments ....
        self.prc_fg_mask = config.get("SPEAKER_SUMMARY_PRC_FG_MASK", None)
        # 0.25 based on experiments ....
        self.prc_speaker_mask = config.get("SPEAKER_SUMMARY_PRC_SPEAKER_MASK", None)

        self.verbose = verbose

    def get_binary_keyframes(self, compressed_frames):
        if self.verbose:
            print(" ... binarizing frames ...")

        all_binary = []
        for idx, compressed_frame in enumerate(compressed_frames):
            # decompress ...
            key_frame = cv2.imdecode(compressed_frame, cv2.IMREAD_COLOR)

            # binarize it ...
            binary_frame = self.ml_binarizer.hysteresis_binarize(key_frame)

            all_binary.append(binary_frame)

            # cv2.imwrite("ZZZ_" + current_lecture.id + "_" + str(idx) + "_raw.png", key_frame)
            # cv2.imwrite("ZZZ_" + current_lecture.id + "_" + str(idx) + "_binary.png", binary_frame)

        return all_binary

    def remove_background(self, binary_image, fg_mask, prc_inner):
        # assuming white CCs for binary image, white fg region in fg mask ...

        # labeling
        cc_labels, count_labels = sci_mes.label(binary_image)

        # finding the size of each CC and how many of these pixels also form part of the mask ...
        cc_index = list(range(0, count_labels + 1))
        cc_sizes = sci_mes.sum(binary_image, cc_labels, cc_index) / 255
        cc_in_mask = sci_mes.sum(fg_mask, cc_labels, cc_index) / 255

        # get proportion inside of the mask
        cc_prc_in_mask = cc_in_mask.copy()
        cc_prc_in_mask[0] = 0
        cc_prc_in_mask[1:] /= cc_sizes[1:]

        ccs_to_keep = cc_prc_in_mask >= prc_inner
        valid_mask = ccs_to_keep[cc_labels]

        result = valid_mask.astype(np.uint8) * 255
        # cv2.imshow("image", binary_image)
        # cv2.imshow("mask", fg_mask)
        # cv2.imshow("result", result)
        # cv2.waitKey()

        # print(sci_mes.sum(binary_image, binary_image, list(range(0, count_labels))))
        return result

    def combine_segment_keyframes(self, segment_all_kf_info, img_size, all_binary, bin_offset):
        segment_binary = np.zeros(img_size, np.uint8)
        for keyframe_idx, spk_bbox in segment_all_kf_info:
            # get next binary from the sequence (these should be sorted by segment and within segment)
            current_binary = all_binary[bin_offset]
            bin_offset += 1

            # remove speaker bbox ...
            if spk_bbox is not None:
                spk_x1, spk_y1, spk_x2, spk_y2 = spk_bbox

                if self.prc_speaker_mask is None:
                    # simply cut the box ...
                    current_binary[spk_y1:spk_y2, spk_x1:spk_x2] = 0
                else:
                    # create a mask for the speaker (or the inverse of the speaker)
                    non_speaker = np.ones(img_size, np.uint8) * 255
                    non_speaker[spk_y1:spk_y2, spk_x1:spk_x2] = 0

                    # keeps entire CCs and removes element outside of the bbox if they intersect it
                    current_binary = self.remove_background(current_binary, non_speaker, 1.0 - self.prc_speaker_mask)

            # print((keyframe_idx, spk_bbox))
            # cv2.imshow("current binary", current_binary)
            # cv2.waitKey()

            # equivalent of OR operation for uint8
            segment_binary = np.maximum(segment_binary, current_binary)

        return segment_binary, bin_offset

    def get_summary(self, video_segment_data, video_keyframes_data, fg_mask):
        # first, binarize key-frames ....
        frame_times, frame_indices, compressed_frames = video_keyframes_data
        all_binary = self.get_binary_keyframes(compressed_frames)

        # to recover the original segment start-end times by interpolate from the know frame times
        # assuming a fixed frame rate on the original video
        times = interp1d(frame_indices, frame_times, fill_value="extrapolate")

        if self.verbose:
            print(" ... generating per segment info ...")

        original_intervals, kfs_per_segment = video_segment_data
        # print(frame_indices)
        # print(kfs_per_segment)

        bin_offset = 0
        keyframes = []
        summary_indices = []
        summary_times = []
        idx_intervals = []
        time_intervals = []

        # for each segment of the video .....
        for int_idx, (segment_start, segment_end) in enumerate(original_intervals):
            # combine segment frames if more than one ...
            # segment_all_kf_info = kfs_per_segment[int_idx]
            segment_binary, bin_offset = self.combine_segment_keyframes(kfs_per_segment[int_idx], fg_mask.shape,
                                                                        all_binary, bin_offset)

            # get abs frame index and and frame time from last binary used ...
            summary_indices.append(frame_indices[bin_offset - 1])
            summary_times.append(frame_times[bin_offset - 1])

            # ... due to sampling error ... the actual frame extracted might be a bit outside of the original interval
            # ... expand the interval to keep it consistent ...
            summ_segment_end = max(segment_end, frame_indices[bin_offset - 1] + 1)
            idx_intervals.append((segment_start, summ_segment_end))
            # ... and compute corresponding times ...
            time_start, time_end = times([segment_start, summ_segment_end])
            time_intervals.append((time_start, time_end))

            # do background removal using mask ...
            if self.prc_fg_mask is not None:
                segment_binary = self.remove_background(segment_binary, fg_mask, self.prc_fg_mask)

            # cv2.imwrite("ZZZ_" + current_lecture.id + "_" + str(int_idx) + "_fg_mask.png", fg_mask)
            # cv2.imwrite("ZZZ_" + current_lecture.id + "_" + str(int_idx) + "_result.png", segment_binary)

            # invert binary ...
            segment_binary = 255 - segment_binary
            keyframes.append(segment_binary)

            # cv2.imshow("binary", segment_binary)
            # cv2.waitKey()

        return keyframes, idx_intervals, time_intervals, summary_indices, summary_times

    def export_summary(self, database, current_lecture, video_segment_data, video_keyframes_data, fg_mask,
                       output_prefix):

        summary_info = self.get_summary(video_segment_data, video_keyframes_data, fg_mask)

        # TODO: this should become a class with 5 fields ... a list of key-frames with info instead of 5 lists ...
        keyframes, idx_intervals, time_intervals, summary_indices, summary_times = summary_info

        if self.verbose:
            print("Saving data to: " + output_prefix)

        KeyframeExporter.Export(output_prefix, database, current_lecture, idx_intervals, time_intervals,
                                summary_indices, summary_times, keyframes)

