# save the openpose data into different videos pickle file,
# based on frames segment infos(segment start and segment end)

import os
import sys

from AM_CommonTools.configuration.configuration import Configuration
from AccessMath.util.misc_helper import MiscHelper
from AccessMath.data.meta_data_DB import MetaDataDB

import pandas as pd

from AccessMath.speaker.data.lecture_pose_segments import LecturePoseSegments
from AccessMath.speaker.data.pose_segment_data import PoseSegmentData

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

    unique_label = config.get("SPEAKER_VALID_ACTIONS")

    dataset_name = config.get("SPEAKER_TRAINING_SET_NAME")

    training_set = database.datasets[dataset_name]

    remove_confidence = config.get("SPEAKER_REMOVE_JOINT_CONFIDENCE")
    normalization_bone = config.get("SPEAKER_NORMALIZATION_BONE")  # pair of norm factor points

    # get the paths to the outputs from previous scripts ....
    output_dir = config.get_str("OUTPUT_PATH")
    action_object_name = config.get_str("SPEAKER_ACTION_MAIN_OBJECT", "speaker")
    action_segment_output_dir = config.get_str("SPEAKER_ACTION_SEGMENT_OUTPUT_DIR", ".")
    segments_output_prefix = output_dir + "/" + action_segment_output_dir + "/" + database.name + "_"

    # the per lecture openpose CSV
    lecture_filename_prefix = output_dir + "/" + config.get_str("OPENPOSE_OUTPUT_DIR_CSV") + "/" + database.name + "_"

    output_segment_dir = output_dir + "/" + config.get("SPEAKER_ACTION_SEGMENT_POSE_DATA_OUTPUT_DIR")
    os.makedirs(output_segment_dir, exist_ok=True)

    # First .... cache all OpenPose CSV data per training lecture ....
    data_per_lecture = {}
    for lecture in training_set:
        lecture_filename = lecture_filename_prefix + lecture.title + ".csv"
        print("Loading data for: " + lecture_filename)

        segments, data = LecturePoseSegments.InitializeFromLectureFile(lecture_filename, normalization_bone,
                                                                       remove_confidence)

        data_per_lecture[lecture.title.lower()] = {
            "segments": segments,
            "data": data
        }

    # read the training frame segments info file
    segment_filename = segments_output_prefix + dataset_name + "_" + action_object_name + ".csv"
    speaker_seg_train = pd.read_csv(segment_filename)  # frame segment info of training data of object speaker
    speaker_seg_train = speaker_seg_train.values

    # Split the OpenPose Data based on the given segments ...
    for vid_name, f_start, f_end, label in speaker_seg_train:
        vid_name = vid_name.lower()
        # print((vid_name, f_start, f_end, label))

        # if label is not in the main 8 labels, omit it
        if label not in unique_label:
            continue

        if not vid_name in data_per_lecture:
            print("Invalid lecture name found: " + vid_name)
            continue

        temp_data = data_per_lecture[vid_name]["data"][f_start:f_end + 1, :]

        temp_pose_segment_data = PoseSegmentData(f_start, f_end, label, temp_data)
        data_per_lecture[vid_name]["segments"].segments.append(temp_pose_segment_data)

    # save to file ...
    for lecture in training_set:
        output_filename = output_segment_dir + "/" + database.name + "_" + lecture.title + ".pickle"
        MiscHelper.dump_save(data_per_lecture[lecture.title.lower()]["segments"], output_filename)

    print("Data Segment Saving Done.")
    return


if __name__ == '__main__':
    main()

