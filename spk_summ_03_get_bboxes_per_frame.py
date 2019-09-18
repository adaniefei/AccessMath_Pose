
import os
import sys

import numpy as np

from AM_CommonTools.configuration.configuration import Configuration
from AccessMath.util.misc_helper import MiscHelper
from AccessMath.data.meta_data_DB import MetaDataDB

from AccessMath.speaker.data.pose_segment_data import PoseSegmentData

from AccessMath.speaker.util.result_recorder import ResultRecorder
from AccessMath.speaker.util.result_reader import ResultReader


def main():
    if len(sys.argv) < 2:
        print("Usage")
        print("\tpython {0:s} config [gt_labels]".format(sys.argv[0]))
        print("\n\tWhere:")
        print("\tgt_lablels:\t(Optional) Set to 1 to use Ground Truth labels instead of predictions")
        return

    # initialization #
    config = Configuration.from_file(sys.argv[1])

    try:
        database = MetaDataDB.from_file(config.get_str("VIDEO_DATABASE_PATH"))
    except:
        print("Invalid AccessMath Database file")
        return

    dataset_name = config.get("SPEAKER_TESTING_SET_NAME")
    testing_set = database.datasets[dataset_name]

    valid_actions = config.get("SPEAKER_VALID_ACTIONS")

    # get the paths to the outputs from previous scripts ....
    output_dir = config.get_str("OUTPUT_PATH")
    output_segment_dir = output_dir + "/" + config.get("SPEAKER_ACTION_SEGMENT_POSE_DATA_OUTPUT_DIR")
    action_class_probabilities_dir = output_dir + "/" + config.get("SPEAKER_ACTION_CLASSIFICATION_PROBABILITIES_DIR")

    output_bboxes_dir = output_dir + "/" + config.get("SPEAKER_ACTION_CLASSIFICATION_BBOXES_DIR")
    os.makedirs(output_bboxes_dir, exist_ok=True)

    remove_confidence = config.get("SPEAKER_REMOVE_JOINT_CONFIDENCE")
    speaker_right_handed = config.get("SPEAKER_IS_RIGHT_HANDED")

    n_joints_body = 25
    n_joints_hand = 21

    if len(sys.argv) >= 3:
        use_ground_truth = int(sys.argv[2]) > 0
    else:
        use_ground_truth = False

    col_name = ['frame_id', ('ground_truth' if use_ground_truth else 'pred_label'),
                'body_xmin', 'body_xmax', 'body_ymin', 'body_ymax',
                'rh_xmin', 'rh_xmax', 'rh_ymin', 'rh_ymax']

    segment_length = config.get_int("SPEAKER_ACTION_SEGMENT_LENGTH")

    # load data + label
    for lecture in testing_set:
        input_filename = output_segment_dir + "/" + database.name + "_" + lecture.title + ".pickle"
        lec_segments = MiscHelper.dump_load(input_filename)

        if use_ground_truth:
            labels = lec_segments.get_all_labels()
        else:
            input_proba_filename = action_class_probabilities_dir + "/" + database.name + "_" + lecture.title + ".csv"
            _, _, labels, _ = ResultReader.read_actions_probabilities_file(input_proba_filename, valid_actions)

        output_filename = output_bboxes_dir + "/" + database.name + "_" + lecture.title + ".csv"
        output_file = ResultRecorder(output_filename)
        output_file.write_headers(col_name)

        # get bbox for skeleton and right hands from all segments
        frames = []
        segment_labels = []
        body_bbox = []
        rh_bbox = []
        for ind in range(0, len(lec_segments.segments)):
            # get the pose data ...
            if not remove_confidence:
                # the data contains confidence ... which needs to be removed at this point ...
                base_pose_data = lec_segments.segments[ind].pose_data

                total_joints = n_joints_body + n_joints_hand * 2
                seg_pose_data = np.zeros((base_pose_data.shape[0], total_joints * 2), dtype=base_pose_data.dtype)

                seg_pose_data[:, ::2] = base_pose_data[:, ::3]
                seg_pose_data[:, 1::2] = base_pose_data[:, 1::3]
            else:
                # confidence has been removed ....
                seg_pose_data = lec_segments.segments[ind].pose_data

            body_features = seg_pose_data[:, 0:n_joints_body * 2]
            if speaker_right_handed:
                # get right hand data
                rh_features = seg_pose_data[:, (n_joints_body + n_joints_hand) * 2:]
            else:
                # use left hand data
                rh_features = seg_pose_data[:, n_joints_body * 2:(n_joints_body + n_joints_hand) * 2]

            # get body bboxes and add to the list ....
            temp_body_bbox = PoseSegmentData.get_bbox_frame_data(body_features, 2)
            body_bbox += temp_body_bbox.tolist()

            # get hand bboxes and add to the list ....
            temp_rh_bbox = PoseSegmentData.get_bbox_frame_data(rh_features, 2)
            rh_bbox += temp_rh_bbox.tolist()

            # add frame range ....
            f_start = lec_segments.segments[ind].frame_start
            f_end = lec_segments.segments[ind].frame_end
            temp_frames = list(range(f_start, f_end + 1))
            frames += temp_frames

            # add label ....
            temp_label = [[labels[ind]] for _ in range(segment_length)]  # remove seg_len, you don't need this
            segment_labels += temp_label

        paras = frames, segment_labels, body_bbox, rh_bbox
        output_file.record_results(paras)


if __name__ == "__main__":
    main()
