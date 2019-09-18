
import os
import sys

from AM_CommonTools.configuration.configuration import Configuration
from AccessMath.data.meta_data_DB import MetaDataDB

from AccessMath.speaker.util.action_segment_sampling import ActionSegmentSampling


def main():
    # usage check
    if len(sys.argv) < 2:
        print("Usage:")
        print("")
        print("\tpython {0:s} config".format(sys.argv[0]))
        print("")
        print("Where")
        print("\tconfig:\tPath to AccessMath configuration file")
        return

    # read the configuration file ....
    config = Configuration.from_file(sys.argv[1])

    try:
        database = MetaDataDB.from_file(config.get_str("VIDEO_DATABASE_PATH"))
    except:
        print("Invalid AccessMath Database file")
        return

    output_dir = config.get_str("OUTPUT_PATH")
    export_prefix = output_dir + "/" + database.output_annotations + "/" + database.name + "_"

    action_object_name = config.get_str("SPEAKER_ACTION_MAIN_OBJECT", "speaker")
    action_segment_length = config.get_int("SPEAKER_ACTION_SEGMENT_LENGTH", 15)
    action_segment_sampling = config.get_int("SPEAKER_ACTION_SEGMENT_SAMPLING_MODE", 2) # MODE!!
    action_segment_tracks = config.get_int("SPEAKER_ACTION_SEGMENT_SAMPLING_TRACKS", 4)
    action_segment_output_dir = config.get_str("SPEAKER_ACTION_SEGMENT_OUTPUT_DIR", ".")

    segments_output_prefix = output_dir + "/" + action_segment_output_dir + "/" + database.name + "_"
    os.makedirs(output_dir + "/" + action_segment_output_dir, exist_ok=True)

    sampler = ActionSegmentSampling(action_segment_sampling, action_segment_length, action_segment_tracks)

    # for each data set ...
    for dataset_name in database.datasets:
        print("Processing data set: " + dataset_name)
        # get segments ...
        all_dataset_segments = []
        for current_lecture in database.datasets[dataset_name]:
            exported_data_filename = export_prefix + current_lecture.title.lower() + "_" + action_object_name + ".csv"
            print(" - input file: " + exported_data_filename)

            if not os.path.exists(exported_data_filename):
                print("\tWARNING: File not found!")
                continue

            # call here the sampler ....
            lecture_title = current_lecture.title.lower()
            lecture_segments = sampler.sample_from_file(exported_data_filename, lecture_title)

            all_dataset_segments += lecture_segments

        # prepare text lines ...
        output_lines = ["lecture_title,frame_start,frame_end,action\n"]
        for segment in all_dataset_segments:
            output_lines.append(",".join([str(value) for value in segment]) + "\n")

        # save segments for dataset ....
        output_filename = segments_output_prefix + dataset_name + "_" + action_object_name + ".csv"
        with open(output_filename, "w") as out_file:
            out_file.writelines(output_lines)

        print(" - data saved to: " + output_filename)

    print("Process complete!")


if __name__ == "__main__":
    main()
