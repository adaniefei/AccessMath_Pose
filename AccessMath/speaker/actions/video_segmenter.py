
import numpy as np

class VideoSegmenter:
    def __init__(self, kf_width, kf_height):
        self.kf_width = kf_width
        self.kf_height = kf_height
        self.spk_bbox_exp_factor = 1.0
        self.min_kf_coverage = 0.10
        # based on grid-search with action segments of 15 frames
        self.erase_merge_gap = 8    # ? seconds
        self.erase_min_size = 6     # ? second
        self.min_split_size = 100   # ? seconds

    def expand_bbox(self, bbox, exp_factor):
        x1, y1, x2, y2 = bbox

        box_w = x2 - x1
        box_h = y2 - y1
        exp_w = (box_w * (exp_factor - 1)) / 2.0
        exp_h = (box_h * (exp_factor - 1)) / 2.0

        # convert to valid boundaries ....
        exp_x1 = max(0, int(x1 - exp_w))
        exp_y1 = max(0, int(y1 - exp_h))
        exp_x2 = min(self.kf_width, int(x2 + 1 + exp_w))
        exp_y2 = min(self.kf_height, int(y2 + 1 + exp_h))

        return exp_x1, exp_y1, exp_x2, exp_y2

    def compute_bboxes_ages(self, bboxes, zero_age, offset_age, exp_factor):
        per_pixel_ages = np.zeros((self.kf_height, self.kf_width), np.int32)
        per_pixel_ages[:, :] = zero_age

        for idx, bbox in enumerate(bboxes):
            exp_x1, exp_y1, exp_x2, exp_y2 = self.expand_bbox(bbox, exp_factor)

            per_pixel_ages[exp_y1:exp_y2, exp_x1:exp_x2] = offset_age + idx

        return per_pixel_ages

    def find_max_pixel_coverage_frame(self, per_pixel_ages, frame_idxs, bboxes):
        offset_age = frame_idxs[0]
        pixels_of_interest = per_pixel_ages >= offset_age
        total_non_minimum = pixels_of_interest.sum()
        best_coverage_idx = None
        best_coverage_total = None
        best_coverage_mask = None
        best_speaker_box = None
        perfect_start_idx = None
        perfect_end_idx = None
        for idx, bbox in reversed(list(enumerate(bboxes))):
            # expand speaker bbox by given factor (and apply boundary constraints)
            exp_x1, exp_y1, exp_x2, exp_y2 = self.expand_bbox(bbox, self.spk_bbox_exp_factor)

            # frame age
            current_frame_idx = frame_idxs[idx]

            # compute coverage ...
            # ... check which pixels have been modified for the last time before this frame ....
            current_coverage = np.logical_and(per_pixel_ages, per_pixel_ages < current_frame_idx)
            max_coverage = current_coverage.sum()
            # ... the ones covered by the current speaker location would not recovered
            current_coverage[exp_y1:exp_y2, exp_x1:exp_x2] = 0

            total_coverage = current_coverage.sum()

            # print((current_frame_idx, max_coverage, total_coverage))
            if perfect_start_idx is not None:
                # it found an ideal key-frame previously ... check for longer ideal segment ...
                if total_coverage == total_non_minimum:
                    # keep extending the interval ...
                    perfect_start_idx = idx
                else:
                    # the "perfect segment" has ended!
                    break
            elif best_coverage_total is None or best_coverage_total < total_coverage:
                # a new best has been found ...
                best_coverage_total = total_coverage
                best_coverage_idx = idx
                best_coverage_mask = current_coverage
                best_speaker_box = (exp_x1, exp_y1, exp_x2, exp_y2)

                if best_coverage_total == total_non_minimum:
                    # perfect coverage found ...
                    perfect_start_idx = idx
                    perfect_end_idx = idx
            elif max_coverage < best_coverage_total or max_coverage == 0:
                # from this point ... the older frames cannot give better coverage than the frame with the best coverage
                # found so far ... stop ...
                break

        if perfect_start_idx is not None:
            # check for middle point of contiguous frames with max coverage
            # note this is not a guarantee of getting a better key-frame but hopefully is better than last from interval
            best_coverage_idx = int((perfect_start_idx + perfect_end_idx) / 2)

        return best_coverage_idx, best_speaker_box, best_coverage_mask, best_coverage_total

    def merge_delete_intervals(self, intervals):
        if len(intervals) == 0:
            return []

        merge_start, merge_end = intervals[0]
        final_intervals = []
        for int_start, int_end in intervals[1:]:
            if int_start - merge_end - 1 <= self.erase_merge_gap:
                # can be merged ... merge!
                merge_end = int_end
            else:
                # cannot be merged ... add previous to result if large enough ...
                if merge_end - merge_start + 1 >= self.erase_min_size:
                    final_intervals.append((merge_start, merge_end))

                # reset current merging interval ...
                merge_start, merge_end = int_start, int_end

        # for last interval ...
        if merge_end - merge_start + 1 >= self.erase_min_size:
            final_intervals.append((merge_start, merge_end))

        return final_intervals

    def split_video(self, erasing_intervals, start_segment, end_segment):
        # base case: the segment is already too short to include ...
        # if end_segment - start_segment + 1 < min_split_size:
        #    return []

        # base case: the segment has no more splitting points ...
        # if len(erasing_intervals) == 0 :
        # if len(erasing_intervals) == 0 or end_segment - start_segment + 1 < min_split_size:
        #    return [(start_segment, end_segment)]

        # greedily choose the longest interval...
        # longest_idx = np.argmax([end_val - start_val for start_val, end_val in erasing_intervals])
        # longest_start, longest_end = erasing_intervals[longest_idx]

        # greedily choose the longest interval... that will not produce "short" segments  ...
        int_lengths = [(end_val - start_val, idx) for idx, (start_val, end_val) in enumerate(erasing_intervals)]
        int_lengths = sorted(int_lengths, reverse=True)
        longest_idx = None
        for int_length, int_idx in int_lengths:
            int_start, int_end = erasing_intervals[int_idx]
            if int_start - start_segment >= self.min_split_size and end_segment - int_end >= self.min_split_size:
                # valid split point found ... stop!
                longest_idx = int_idx
                break

        if longest_idx is None:
            # no valid split point was found .. keep
            return [(start_segment, end_segment)]
        else:
            # recursively split ...
            longest_start, longest_end = erasing_intervals[longest_idx]
            part_a = self.split_video(erasing_intervals[:longest_idx], start_segment, longest_start - 1)
            part_b = self.split_video(erasing_intervals[longest_idx + 1:], longest_end + 1, end_segment)

            return part_a + part_b

    def find_keyframes_per_split(self, video_splits, segments, segment_actions, frame_idxs, frame_body_bboxes,
                                 frame_rh_bboxes):
        selected_keyframes = []
        for split_start, split_end in video_splits:
            frame_start = segments[split_start][0]
            frame_end = segments[split_end][1]
            # print("")
            # print((frame_start, frame_end))

            split_actions = segment_actions[split_start:split_end + 1]
            writing_intervals = VideoSegmenter.find_action_intervals(split_actions, "write")
            out_intervals = VideoSegmenter.find_action_intervals(split_actions, "out")

            # find any intervals where speaker is out after last writing event ...
            # ... remember these intervals are based on local offsets ....
            if len(writing_intervals) > 0:
                # get the offset of the last writing interval
                last_writing_offset = writing_intervals[-1][1]
            else:
                # segment without writing events ??? almost any frame can be a key-frame
                last_writing_offset = 0

            # if the speaker is out at least once ... and the last time is out is after the last writing event
            # ... then key-frame can be easily chosen in this region ...
            if len(out_intervals) > 0 and last_writing_offset < out_intervals[-1][0]:
                # ... the mid point of last out interval ...
                kf_segment_offset = int((out_intervals[-1][0] + out_intervals[-1][1]) / 2)

                kf_segment_start, kf_segment_end = segments[split_start + kf_segment_offset]
                kf_segment_frame_idx = int((kf_segment_start + kf_segment_end) / 2)

                # full key-frame, no restricted bounding box ...
                selected_keyframes.append([(kf_segment_frame_idx, None)])

                # print((segment_actions[split_start + kf_segment_offset], kf_segment_frame_idx))
            else:
                # get the segment info ... (frames and bboxes)
                segment_mask = np.logical_and(frame_start <= frame_idxs, frame_idxs <= frame_end)

                split_frame_idxs = frame_idxs[segment_mask]
                split_body_bboxes = frame_body_bboxes[segment_mask]
                split_rh_bboxes = frame_rh_bboxes[segment_mask]

                # body_ages = compute_bboxes_ages(split_body_bboxes, 0, frame_start, kf_w, kf_h, speaker_expand)
                hand_ages = self.compute_bboxes_ages(split_rh_bboxes, 0, frame_start, self.spk_bbox_exp_factor)

                # TODO: set a maximum number of key-frames per segment?
                total_to_cover = (hand_ages > 0).sum()

                # iteratively chose keyframes ...
                segment_keyframes = []
                while (hand_ages > 0).sum() > 0:
                    # find next frame with the best coverage ...
                    kf_info = self.find_max_pixel_coverage_frame(hand_ages, split_frame_idxs, split_body_bboxes)
                    frame_offset, kf_speaker_box, kf_coverage, kf_total_coverage = kf_info

                    kf_segment_frame_idx = split_frame_idxs[frame_offset]

                    # check if best frame has coverage, otherwise, extraction cannot continue
                    if kf_total_coverage == 0 or (kf_total_coverage / total_to_cover) < self.min_kf_coverage:
                        break

                    # add to key-frames
                    segment_keyframes.append((kf_segment_frame_idx, kf_speaker_box))

                    # remove covered
                    # cv2.imshow("before", cv2.resize((hand_ages > 0).astype(np.uint8) * 255, (960, 540)))
                    hand_ages[kf_coverage] = 0
                    """
                    tempo_img = np.zeros((kf_h, kf_w, 3), np.uint8)
                    tempo_img[:, :, 0] = (hand_ages > 0).astype(np.uint8) * 255
                    tempo_img[:, :, 1] = tempo_img[:, :, 0].copy()
                    tempo_img[:, :, 2] = tempo_img[:, :, 0].copy()
                    print(kf_speaker_box)
                    print(kf_segment_frame_idx)
                    cv2.rectangle(tempo_img, (kf_speaker_box[0], kf_speaker_box[1]), (kf_speaker_box[2], kf_speaker_box[3]),
                                  (0,0,255), thickness=3)
                    cv2.imshow("after", cv2.resize(tempo_img, (960, 540)))
                    cv2.waitKey()
                    print((segment_keyframes, (hand_ages > 0).sum()))
                    """

                # sort the selected key-frames
                segment_keyframes = sorted(segment_keyframes, key=lambda x: x[0])

                selected_keyframes.append(segment_keyframes)

                # print(segment_keyframes)

            # print(frame_idxs)
        return selected_keyframes

    def get_keyframes(self, pred_actions, segments, frame_idxs, body_bboxes, rh_bboxes):
        # TODO: this refinement procedure does not work
        # refine_action_predictions(pred_actions, prob_actions, action_stats, inv_action_idx, gt_actions)

        # Find the segments ...

        print("predicted")
        erase_intervals = VideoSegmenter.find_action_intervals(pred_actions, "erase")
        print(len(erase_intervals))
        print(erase_intervals)
        print("")

        # erase_intervals = segmenter.merge_delete_intervals(erase_intervals, erase_merge_gap, erase_min_size)
        erase_intervals = self.merge_delete_intervals(erase_intervals)
        print("merged-deleted")
        print(len(erase_intervals))
        print(erase_intervals)
        print("")

        print("... video splits ...")
        # video_splits = segmenter.split_video(erase_intervals, 0, len(segments) - 1, min_split_size)
        video_splits = self.split_video(erase_intervals, 0, len(segments) - 1)
        print(len(video_splits))
        print(video_splits)
        splits_frames = [(segments[start][0], segments[end][1]) for start, end in video_splits]
        print(splits_frames)
        print("")

        # print("ground truth")
        # erase_intervals = find_action_intervals(gt_actions, "erase")
        # print(len(erase_intervals))
        # print(erase_intervals)
        # print("-----")

        # get video segment key-frames
        video_keyframes = self.find_keyframes_per_split(video_splits, segments, pred_actions, frame_idxs, body_bboxes,
                                                        rh_bboxes)

        return splits_frames, video_keyframes

    @staticmethod
    def find_action_intervals(actions, action_name):
        int_start = None
        pos = 0
        intervals = []
        # check the array ..
        while pos < len(actions):
            if actions[pos] == action_name:
                # action found ... check if first
                if int_start is None:
                    # record interval start ...
                    int_start = pos
            else:
                # different action ... check if an interval was being processed
                if int_start is not None:
                    # the interval finished in the last element
                    intervals.append((int_start, pos - 1))
                    int_start = None

            pos += 1

        # in case an interval finishes at the end of the actions array ...
        if int_start is not None:
            intervals.append((int_start, pos - 1))

        return intervals

    @staticmethod
    def FromConfig(config, kf_width, kf_height):

        segmenter = VideoSegmenter(kf_width, kf_height)
        segmenter.spk_bbox_exp_factor = config.get_float("SPEAKER_SEGMENT_BBOX_EXPANSION_FACTOR", 1.0)
        segmenter.min_kf_coverage = config.get_float("SPEAKER_SEGMENT_MIN_KF_COVERAGE", 0.10)

        segmenter.erase_merge_gap = config.get_int("SPEAKER_SEGMENT_ERASE_MERGE_GAP", 8)
        segmenter.erase_min_size = config.get_int("SPEAKER_SEGMENT_ERASE_MIN_SIZE", 6)
        segmenter.min_split_size = config.get_int("SPEAKER_SEGMENT_MIN_SPLIT_SIZE", 100)  # ? seconds

        return segmenter


# DEPRECATED CODE!
"""
def tri_gram_probabilty(first, second, third, action_stats):
    count_tri = action_stats[first, second, third]
    count_hist = action_stats[first, second].sum()

    return count_tri / count_hist

def per_action_sequence_prob(offset, context, pred_actions, prob_actions, action_stats, inv_action_idx):
    # assuming tri-grams ... compute probabilities for different actions on a given sequence
    # where the changing element is the middle action (action 3)
    # [act_1] [act_2] {action_3} [act_4] [act_5]
    # act_1 = inv_action_idx[pred_actions[offset - 2]]
    # act_2 = inv_action_idx[pred_actions[offset - 1]]
    # act_4 = inv_action_idx[pred_actions[offset + 1]]
    # act_5 = inv_action_idx[pred_actions[offset + 2]]
    
    action_idxs = [inv_action_idx[pred_actions[offset + act_idx]] for act_idx in range(-context, context + 1)]

    per_action_prob = {}
    highest = None
    for action_label in inv_action_idx:
        # act_3 = inv_action_idx[action_label]
        action_idxs[context] = inv_action_idx[action_label]

        seq_prob = 1.0
        context * 2 - 1

        for trigram_idx in range(context * 2 - 1):
            seq_prob *= tri_gram_probabilty(action_idxs[trigram_idx], action_idxs[trigram_idx + 1],
                                            action_idxs[trigram_idx + 2], action_stats)

        # left = tri_gram_probabilty(act_1, act_2, act_3, action_stats)
        # mid = tri_gram_probabilty(act_2, act_3, act_4, action_stats)
        # right = tri_gram_probabilty(act_3, act_4, act_5, action_stats)

        # seq_prob = left * mid * right
        combined = seq_prob * prob_actions[action_label][offset]

        per_action_prob[action_label] = combined
        if highest is None or combined > per_action_prob[highest]:
            highest = action_label

        # print((action_label, prob_actions[action_label][offset], seq_prob, combined))

    # print(highest)

    return per_action_prob, highest

def refine_action_predictions(pred_actions, prob_actions, action_stats, inv_action_idx, gt_actions):
    # (n before and n after)
    context = 2
    # debug: check for error rate?
    count_errors = 0
    border_errors = 0
    tempo_probs = []
    okay_probs = []
    wrong_not_changed = 0
    wrong_changed_right = 0
    wrong_changed_wrong = 0
    right_not_changed = 0
    right_changed= 0
    for idx in range(len(pred_actions)):
        # prob_actions[pred_actions[idx]][idx] < 0.5
        if (context <= idx < len(pred_actions) - context):
            action_context = []
            for context_idx in range(1, context + 1):
                action_context.append(pred_actions[idx - context_idx])
                action_context.append(pred_actions[idx + context_idx])

            if pred_actions[idx] not in action_context:
                print("")
                print(pred_actions[idx - 2:idx + 3])
                print(gt_actions[idx - 2:idx + 3])
                per_action_prob, highest = per_action_sequence_prob(idx, context, pred_actions, prob_actions, action_stats,
                                                                    inv_action_idx)
                print(highest)
            else:
                highest = None
        else:
            highest = None

        if pred_actions[idx] != gt_actions[idx]:
            count_errors += 1

            if 1 <= idx < len(pred_actions) - 1:
                # GT previous and next actions are different and ...
                # GT current is same as either previous or next and ..
                # Prediction for previous and Prediction for next are both correct ...
                # Prediction for current is same as either previous or next ..
                # .... but different from GT value ... then it is a border confusion
                if (gt_actions[idx -1] != gt_actions[idx + 1] and
                    (gt_actions[idx] == gt_actions[idx - 1] or gt_actions[idx] == gt_actions[idx + 1]) and
                    (pred_actions[idx - 1] == gt_actions[idx -1] and pred_actions[idx + 1] == gt_actions[idx + 1]) and
                    (pred_actions[idx] == pred_actions[idx - 1] or pred_actions[idx] == pred_actions[idx + 1])):
                    border_errors += 1

            # print([prob_actions[valid_action][idx] for valid_action in inv_action_idx])
            # print((prob_actions[pred_actions[idx]][idx], prob_actions[gt_actions[idx]][idx]))
            tempo_probs.append([prob_actions[pred_actions[idx]][idx], prob_actions[gt_actions[idx]][idx]])

            if highest is not None:
                if highest != pred_actions[idx]:
                    # it would be changed ...
                    if highest == gt_actions[idx]:
                        # now is correct (the quantity we want to maximize
                        wrong_changed_right += 1
                    else:
                        # annoying, but makes no difference overall
                        wrong_changed_wrong += 1
                else:
                    # it would not be changed ...
                    wrong_not_changed += 1

        else:
            okay_probs.append([prob_actions[pred_actions[idx]][idx], prob_actions[gt_actions[idx]][idx]])

            if highest is not None:
                if highest != pred_actions[idx]:
                    # it would be changed ... for bad ...
                    right_changed += 1
                else:
                    # it would not be changed ... desirable ...
                    right_not_changed += 1

        if highest is not None:
            # print((pred_actions[idx], highest))
            # pred_actions[idx] = highest
            pass

    tempo_probs = np.array(tempo_probs)
    okay_probs = np.array(okay_probs)

    print((count_errors, border_errors, len(pred_actions)))
    print("error Mean and STDev")
    print(tempo_probs.mean(axis=0))
    print(tempo_probs.std(axis=0))
    print("correct Mean and STDev")
    print(okay_probs.mean(axis=0))
    print(okay_probs.std(axis=0))

    print("")
    print("Wrong Not Changed: {0:d}".format(wrong_not_changed))
    print("Wrong changed Wrong: {0:d}".format(wrong_changed_wrong))
    print("Wrong changed Right: {0:d}".format(wrong_changed_right))
    print("Right Not Changed: {0:d}".format(right_not_changed))
    print("Right Changed Wrong: {0:d}".format(right_changed))
    print("")
    print("")
"""