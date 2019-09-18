
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
    temporal_segments_dir = output_dir + "/" + config.get("SPEAKER_ACTION_TEMPORAL_SEGMENTS_DIR")
    keyframes_dir = output_dir + "/" + config.get("SPEAKER_ACTION_KEYFRAMES_DIR")
    os.makedirs(keyframes_dir, exist_ok=True)

    dataset_name = config.get("SPEAKER_TESTING_SET_NAME")
    testing_set = database.datasets[dataset_name]

    for current_lecture in testing_set:
        # read segment data ....
        input_filename = temporal_segments_dir + "/" + database.name + "_" + current_lecture.title + ".pickle"
        video_segment_data = MiscHelper.dump_load(input_filename)

        # key-frames that must be extracted from video ...
        segments, keyframes_per_segment = video_segment_data
        all_keyframes = []
        for segment_keyframes in keyframes_per_segment:
            all_keyframes += [keyframe_idx for keyframe_idx, bbox in segment_keyframes]

        print("")
        print("processing: " + current_lecture.title)
        # print(all_keyframes)

        # the simple frame sampling worker ..
        worker = SimpleFrameSampler()

        # main video file names
        m_videos = [config.get_str("VIDEO_FILES_PATH") + "/" + video["path"] for video in current_lecture.main_videos]

        # execute the actual process ....
        processor = SequentialVideoSampler(m_videos, all_keyframes)

        if "forced_width" in current_lecture.parameters:
            processor.force_resolution(current_lecture.parameters["forced_width"],
                                       current_lecture.parameters["forced_height"])
        processor.doProcessing(worker, 0, True)  # 0

        sampled_frame_data = worker.frame_times, worker.frame_indices, worker.compressed_frames

        # save results
        keyframes_data_filename = keyframes_dir + "/" + database.name + "_" + current_lecture.title + ".pickle"
        MiscHelper.dump_save(sampled_frame_data, keyframes_data_filename)



if __name__ == "__main__":
    main()
