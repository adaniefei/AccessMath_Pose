
import os
import sys

from AM_CommonTools.configuration.configuration import Configuration
from AccessMath.util.misc_helper import MiscHelper
from AccessMath.data.meta_data_DB import MetaDataDB

from AccessMath.preprocessing.video_processor.sequential_video_sampler import SequentialVideoSampler
from AccessMath.preprocessing.video_worker.simple_frame_sampler import SimpleFrameSampler


def main():
    # usage check
    if len(sys.argv) < 2:
        print("Usage:")
        print("")
        print("\tpython {0:s} config [dataset]".format(sys.argv[0]))
        print("")
        print("Where")
        print("\tconfig:\tPath to AccessMath configuration file")
        print("\tdataset:\tDataset to run (Default= Training)")
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
    os.makedirs(video_metadata_dir, exist_ok=True)

    dataset_name = config.get("SPEAKER_TESTING_SET_NAME")
    testing_set = database.datasets[dataset_name]

    for current_lecture in testing_set:
        print("")
        print("processing: " + current_lecture.title)
        # print(all_keyframes)

        # the simple frame sampling worker ..
        worker = SimpleFrameSampler()

        # main video file names
        m_videos = [config.get_str("VIDEO_FILES_PATH") + "/" + video["path"] for video in current_lecture.main_videos]

        video_info = {}
        if "forced_width" in current_lecture.parameters:
            video_info["width"] = current_lecture.parameters["forced_width"]
            video_info["height"] = current_lecture.parameters["forced_height"]
        else:
            # execute the actual process ....
            processor = SequentialVideoSampler(m_videos, [0])
            processor.doProcessing(worker, 0, True)  # 0

            video_info["width"] = worker.width
            video_info["height"] = worker.height

        output_filename = video_metadata_dir + "/" + database.name + "_" + current_lecture.title + ".pickle"
        MiscHelper.dump_save(video_info, output_filename)


if __name__ == "__main__":
    main()
