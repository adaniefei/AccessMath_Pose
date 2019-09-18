
import os
import sys
import cv2

from AM_CommonTools.configuration.configuration import Configuration
from AccessMath.util.misc_helper import MiscHelper
from AccessMath.data.meta_data_DB import MetaDataDB

from AccessMath.speaker.actions.summary_generator import SummaryGenerator

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

    # inputs / output paths
    output_dir = config.get_str("OUTPUT_PATH")
    temporal_segments_dir = output_dir + "/" + config.get("SPEAKER_ACTION_TEMPORAL_SEGMENTS_DIR")
    keyframes_dir = output_dir + "/" + config.get("SPEAKER_ACTION_KEYFRAMES_DIR")
    fg_mask_dir = output_dir + "/" + config.get_str("SPEAKER_FG_ESTIMATION_MASK_DIR")

    summaries_dir = output_dir + "/" + database.output_summaries
    os.makedirs(summaries_dir, exist_ok=True)
    summary_prefix = summaries_dir + "/" + config.get_str("SPEAKER_SUMMARY_PREFIX") + "_" + database.name + "_"

    # current dataset ....
    dataset_name = config.get("SPEAKER_TESTING_SET_NAME")
    testing_set = database.datasets[dataset_name]

    print("... preparing summary generator ...")
    summ_generator = SummaryGenerator(config)


    for current_lecture in testing_set:
        print("")
        print("Processing: " + current_lecture.title)

        # get all inputs ....

        # read segment data ....
        segments_data_filename = temporal_segments_dir + "/" + database.name + "_" + current_lecture.title + ".pickle"
        video_segment_data = MiscHelper.dump_load(segments_data_filename)

        # read key-frames data ...
        keyframes_data_filename = keyframes_dir + "/" + database.name + "_" + current_lecture.title + ".pickle"
        video_keyframes_data = MiscHelper.dump_load(keyframes_data_filename)

        # read mask data ...
        fg_mask_filename = fg_mask_dir + "/" + database.name + "_" + current_lecture.title + ".pickle"
        fg_mask_png = MiscHelper.dump_load(fg_mask_filename)
        fg_mask = cv2.imdecode(fg_mask_png, cv2.IMREAD_GRAYSCALE)

        output_prefix = summary_prefix + current_lecture.title.lower()

        summ_generator.export_summary(database, current_lecture, video_segment_data, video_keyframes_data, fg_mask,
                                      output_prefix)




if __name__ == "__main__":
    main()

