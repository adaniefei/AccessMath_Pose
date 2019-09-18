
import os
import sys

from AM_CommonTools.configuration.configuration import Configuration
from AccessMath.util.misc_helper import MiscHelper
from AccessMath.data.meta_data_DB import MetaDataDB

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

    dataset_name = config.get("SPEAKER_TESTING_SET_NAME")
    testing_set = database.datasets[dataset_name]

    remove_confidence = config.get("SPEAKER_REMOVE_JOINT_CONFIDENCE")
    normalization_bone = config.get("SPEAKER_NORMALIZATION_BONE")  # pair of norm factor points

    # get the paths to the outputs from previous scripts ....
    output_dir = config.get_str("OUTPUT_PATH")

    # the per lecture openpose CSV
    lecture_filename_prefix = output_dir + "/" + config.get_str("OPENPOSE_OUTPUT_DIR_CSV") + "/" + database.name + "_"

    output_segment_dir = output_dir + "/" + config.get("SPEAKER_ACTION_SEGMENT_POSE_DATA_OUTPUT_DIR")
    os.makedirs(output_segment_dir, exist_ok=True)

    segment_length = config.get_int("SPEAKER_ACTION_SEGMENT_LENGTH")

    for lecture in testing_set:
        lecture_filename = lecture_filename_prefix + lecture.title + ".csv"
        print("Loading data for: " + lecture_filename)

        # get the corresponding data for this lecture ...
        lec_segments, lecture_data = LecturePoseSegments.InitializeFromLectureFile(lecture_filename, normalization_bone,
                                                                                   remove_confidence)

        # sequential sampling for pose segments
        vid_len = lecture_data.shape[0]
        for ind in range(0, int(vid_len / segment_length)):
            f_start = ind * segment_length
            f_end = f_start + segment_length - 1
            temp_data = lecture_data[f_start:f_end + 1, :]

            lec_segments.segments.append(PoseSegmentData(f_start, f_end, None, temp_data))

        # save ....
        output_filename = output_segment_dir + "/" + database.name + "_" + lecture.title + ".pickle"
        MiscHelper.dump_save(lec_segments, output_filename)



if __name__ == "__main__":
    main()
