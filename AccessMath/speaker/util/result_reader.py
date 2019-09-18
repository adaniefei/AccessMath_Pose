
import numpy as np

class ResultReader:
    @staticmethod
    def read_actions_probabilities_file(filename, valid_actions):
        with open(filename, "r") as in_file:
            all_lines = in_file.readlines()

        # process header ...
        headers = all_lines[0].strip().split(",")
        header_idx = {value: idx for idx, value in enumerate(headers)}

        # now extract the data ...
        segments = []
        gt_actions = []
        pred_actions = []
        prob_actions = {action_label: [] for action_label in valid_actions}

        for line in all_lines[1:]:
            parts = line.strip().split(",")
            if len(parts) == len(headers):
                # read segment ...
                segment_start = int(parts[header_idx["frame_start"]])
                segment_end = int(parts[header_idx["frame_end"]])
                segments.append((segment_start, segment_end))

                # action label(s) ...
                action_label = parts[header_idx["prediction"]]
                pred_actions.append(action_label)

                if "ground_truth" in header_idx:
                    action_label = parts[header_idx["ground_truth"]]
                    gt_actions.append(action_label)

                # probabilities ...
                for action_label in valid_actions:
                    label_prob = float(parts[header_idx[action_label]])
                    prob_actions[action_label].append(label_prob)

        segments = np.array(segments)

        if len(gt_actions) == 0:
            gt_actions = None

        for action_label in valid_actions:
            prob_actions[action_label] = np.array(prob_actions[action_label])

        return segments, gt_actions, pred_actions, prob_actions

    @staticmethod
    def read_bbox_file(filename, use_ground_truth=False):
        with open(filename, "r") as in_file:
            all_lines = in_file.readlines()

        # process header ...
        headers = all_lines[0].strip().split(",")
        header_idx = {value: idx for idx, value in enumerate(headers)}

        if use_ground_truth:
            label_attribute = "ground_truth"
        else:
            label_attribute = "pred_label"

        # now extract the data ...
        # segments = []
        frame_idxs = []
        actions = []
        body_bboxes = []
        right_hand_bboxes = []

        for line in all_lines[1:]:
            parts = line.strip().split(",")
            if len(parts) == len(headers):
                # read segment ...
                # segment_start = int(parts[header_idx["frame_start"]])
                # segment_end = int(parts[header_idx["frame_end"]])
                # segments.append((segment_start, segment_end))
                frame_id = int(parts[header_idx["frame_id"]])
                frame_idxs.append(frame_id)

                # action label ...
                action_label = parts[header_idx[label_attribute]]
                actions.append(action_label)

                # body bbox
                b_x1 = float(parts[header_idx["body_xmin"]])
                b_y1 = float(parts[header_idx["body_ymin"]])
                b_x2 = float(parts[header_idx["body_xmax"]])
                b_y2 = float(parts[header_idx["body_ymax"]])
                body_bboxes.append((b_x1, b_y1, b_x2, b_y2))

                # right hand
                rh_x1 = float(parts[header_idx["rh_xmin"]])
                rh_y1 = float(parts[header_idx["rh_ymin"]])
                rh_x2 = float(parts[header_idx["rh_xmax"]])
                rh_y2 = float(parts[header_idx["rh_ymax"]])
                right_hand_bboxes.append((rh_x1, rh_y1, rh_x2, rh_y2))

        # segments = np.array(segments)
        frame_idxs = np.array(frame_idxs)
        actions = np.array(actions)
        body_bboxes = np.array(body_bboxes)
        right_hand_bboxes = np.array(right_hand_bboxes)

        return frame_idxs, actions, body_bboxes, right_hand_bboxes
