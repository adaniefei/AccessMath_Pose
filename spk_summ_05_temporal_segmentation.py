
import os
import sys

from AM_CommonTools.configuration.configuration import Configuration
from AccessMath.util.misc_helper import MiscHelper
from AccessMath.data.meta_data_DB import MetaDataDB

from AccessMath.speaker.actions.video_segmenter import VideoSegmenter
from AccessMath.speaker.util.result_reader import ResultReader


def main():
    # usage check
    if len(sys.argv) < 2:
        print("Usage:")
        print("")
        print("\tpython {0:s} config [gt_labels]".format(sys.argv[0]))
        print("")
        print("Where")
        print("\tconfig:\tPath to AccessMath configuration file")
        print("\tgt_labels:\tuse ground truth action labels (Default= False)")
        return

    # read the configuration file ....
    config = Configuration.from_file(sys.argv[1])

    try:
        database = MetaDataDB.from_file(config.get_str("VIDEO_DATABASE_PATH"))
    except:
        print("Invalid AccessMath Database file")
        return

    output_dir = config.get_str("OUTPUT_PATH")
    video_metadata_dir = output_dir + "/" + config.get_str("SPEAKER_ACTION_VIDEO_META_DATA_DIR")
    action_class_probabilities_dir = output_dir + "/" + config.get("SPEAKER_ACTION_CLASSIFICATION_PROBABILITIES_DIR")
    output_bboxes_dir = output_dir + "/" + config.get("SPEAKER_ACTION_CLASSIFICATION_BBOXES_DIR")
    temporal_segments_dir = output_dir + "/" + config.get("SPEAKER_ACTION_TEMPORAL_SEGMENTS_DIR")
    os.makedirs(temporal_segments_dir, exist_ok=True)

    dataset_name = config.get("SPEAKER_TESTING_SET_NAME")
    testing_set = database.datasets[dataset_name]

    valid_actions = config.get("SPEAKER_VALID_ACTIONS")

    if len(sys.argv) >= 3:
        use_ground_truth = int(sys.argv[2]) > 0
    else:
        use_ground_truth = False

    for current_lecture in testing_set:
        info_filename = video_metadata_dir + "/" + database.name + "_" + current_lecture.title + ".pickle"
        proba_filename = action_class_probabilities_dir + "/" + database.name + "_" + current_lecture.title + ".csv"

        video_info = MiscHelper.dump_load(info_filename)

        segmenter = VideoSegmenter.FromConfig(config, video_info["width"], video_info["height"])

        # read label data ....
        prob_info = ResultReader.read_actions_probabilities_file(proba_filename, valid_actions)
        segments, gt_actions, pred_actions, prob_actions = prob_info

        # read bbox data ...
        bbox_filename = output_bboxes_dir + "/" + database.name + "_" + current_lecture.title + ".csv"
        frame_idxs, frame_actions, body_bboxes, rh_bboxes = ResultReader.read_bbox_file(bbox_filename, use_ground_truth)

        # (splits_frames, video_keyframes)
        video_data = segmenter.get_keyframes(pred_actions, segments, frame_idxs, body_bboxes, rh_bboxes)

        print("")
        print("video key_frames")
        print(video_data[0])
        print(video_data[1])
        print("")

        output_filename = temporal_segments_dir + "/" + database.name + "_" + current_lecture.title + ".pickle"
        MiscHelper.dump_save(video_data, output_filename)

if __name__ == "__main__":
    main()
