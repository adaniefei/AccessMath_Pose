
import cv2
import numpy as np

class ForegroundEstimator:
    def __init__(self, width, height, speaker_exp_factor, min_mask_frames, mask_exp_radius):
        self.width = width
        self.height = height
        self.speaker_exp_factor = speaker_exp_factor
        self.min_mask_frames = min_mask_frames
        self.mask_exp_radius = mask_exp_radius

    def recover_missing_bboxes(self, frames, bboxes):
        rec_bboxes = bboxes.copy()

        main_pos = 0
        while main_pos < bboxes.shape[0]:
            x1, y1, x2, y2 = bboxes[main_pos]
            if x1 < 0:
                # missing bbox identified ...
                # find the next good bbox ...
                second_pos = main_pos + 1
                while second_pos < bboxes.shape[0] and bboxes[second_pos][0] < 0:
                    second_pos += 1

                # check if
                #  - it didn't reach the end ...
                #  - and is not a gap at the beginning
                #  - and if the gap is contiguous
                if ((second_pos < bboxes.shape[0]) and (main_pos > 0) and
                    (second_pos - main_pos + 1 == frames[second_pos] - frames[main_pos - 1])):
                    # interpolate between last known box and next known box ...
                    prev_bbox = bboxes[main_pos - 1]
                    next_bbox = bboxes[second_pos]
                    # print((main_pos, second_pos, frames[main_pos - 1:second_pos + 1], bboxes[second_pos]))
                    for inner_frame_idx in range(main_pos, second_pos):
                        w = (inner_frame_idx - main_pos + 1.0) / (second_pos - main_pos + 1.0)

                        current_box = next_bbox * w + prev_bbox * (1.0 - w)
                        rec_bboxes[inner_frame_idx] = current_box

                    pass

                main_pos = second_pos + 1
            else:
                # just keep moving ...
                main_pos += 1

        return rec_bboxes

    def get_bboxes_heat_map(self, bboxes):
        heat_map = np.zeros((self.height, self.width), np.float64)
        count = 0
        for x1, y1, x2, y2 in bboxes:
            if x1 < 0:
                # skip this bbox
                count += 1
                continue

            box_w = x2 - x1
            box_h = y2 - y1
            exp_w = (box_w * (self.speaker_exp_factor - 1)) / 2.0
            exp_h = (box_h * (self.speaker_exp_factor - 1)) / 2.0

            # convert to valid boundaries ....
            x1 = max(0, int(x1 - exp_w))
            y1 = max(0, int(y1 - exp_h))
            x2 = min(self.width, int(x2 + 1 + exp_w))
            y2 = min(self.height, int(y2 + 1 + exp_h))

            heat_map[y1:y2, x1:x2] += 1

        print(" - Boxes ommited: {0:.2f} ({1:d} of {2:d})".format(count * 100.0 / bboxes.shape[0], count,
                                                                  bboxes.shape[0]))

        return heat_map

    def get_action_mask(self, frame_idxs, actions, body_bboxes, right_hand_bboxes, action_name):
        relevant_idx = np.nonzero(actions == action_name)[0]

        # interpolate missing boxes on contiguous segments ...
        action_frame_idxs = frame_idxs[relevant_idx]
        action_rh_bboxes = right_hand_bboxes[relevant_idx]
        action_rh_bboxes = self.recover_missing_bboxes(action_frame_idxs, action_rh_bboxes)

        body_heat_map = self.get_bboxes_heat_map(body_bboxes[relevant_idx])
        hand_heat_map = self.get_bboxes_heat_map(action_rh_bboxes)
        # visualize_heat_map(body_heat_map, "body " + action_name, (960, 540))
        # visualize_heat_map(hand_heat_map, "right hand " + action_name, (960, 540))

        return body_heat_map, hand_heat_map

    def visualize_heat_map(self, heat_map, window_name, window_size):
        vis_heat_map = ((heat_map / heat_map.max()) * 255).astype(np.uint8)
        cv2.imshow(window_name, cv2.resize(vis_heat_map, window_size))
        # cv2.imwrite(window_name + ".png", vis_heat_map)

    def expand_mask(self, bin_image, close_radius):
        struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(close_radius, close_radius))
        # return cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE, struct_elem)
        return cv2.morphologyEx(bin_image, cv2.MORPH_DILATE, struct_elem)

    def get_mask(self, frame_idxs, actions, body_bboxes, right_hand_bboxes, debug_name=None):
        # 'write', 'erase'
        w_body_map, w_hand_map = self.get_action_mask(frame_idxs, actions, body_bboxes, right_hand_bboxes, 'write')
        e_body_map, e_hand_map = self.get_action_mask(frame_idxs, actions, body_bboxes, right_hand_bboxes, 'erase')

        if debug_name is not None:
            self.visualize_heat_map(w_hand_map, debug_name, (int(self.width / 2), int(self.height / 2)))

        # combined_hand_map = write_hand_map + erase_hand_map
        # combined_hand_map = (write_hand_map > 5)
        combined_hand_map = (w_hand_map + e_hand_map > self.min_mask_frames).astype(np.uint8) * 255
        combined_hand_map = self.expand_mask(combined_hand_map, self.mask_exp_radius)

        return combined_hand_map
