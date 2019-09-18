
# Get features from every training videos and save them in corresponding
# pickle files.

import os
import sys

from AM_CommonTools.configuration.configuration import Configuration
from AccessMath.util.misc_helper import MiscHelper
from AccessMath.data.meta_data_DB import MetaDataDB

from AccessMath.speaker.actions.pose_feature_extractor import PoseFeatureExtractor


def main():
    if len(sys.argv) < 2:
        print("Usage")
        print("\tpython {0:s} config".format(sys.argv[0]))
        return

    # initialization #
    config = Configuration.from_file(sys.argv[1])

    try:
        database = MetaDataDB.from_file(config.get_str("VIDEO_DATABASE_PATH"))
    except:
        print("Invalid AccessMath Database file")
        return

    # get paths and other configuration parameters ....
    output_dir = config.get_str("OUTPUT_PATH")
    output_segment_dir = output_dir + "/" + config.get("SPEAKER_ACTION_SEGMENT_POSE_DATA_OUTPUT_DIR")

    dataset_name = config.get("SPEAKER_TRAINING_SET_NAME")
    training_set = database.datasets[dataset_name]

    # prepare the feature extractor ...
    feature_points = config.get("SPEAKER_ACTION_FEATURE_POINTS")
    segment_length = config.get_int("SPEAKER_ACTION_SEGMENT_LENGTH", 15)
    feat_extractor = PoseFeatureExtractor(feature_points, segment_length)

    features_dir = output_dir + "/" + config.get("SPEAKER_ACTION_FEATURES_DIR")
    os.makedirs(features_dir, exist_ok=True)

    # for each file ... get features ...
    for lecture in training_set:
        input_filename = output_segment_dir + "/" + database.name + "_" + lecture.title + ".pickle"
        output_filename = features_dir + "/" + database.name + "_" + lecture.title + ".pickle"

        lecture_pose_segments = MiscHelper.dump_load(input_filename)

        vid_data = feat_extractor.get_feature_dataset(lecture_pose_segments)

        MiscHelper.dump_save(vid_data, output_filename)

    return


if __name__ == '__main__':
    main()

