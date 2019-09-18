
import os
import sys
import cv2


from AM_CommonTools.configuration.configuration import Configuration
from AccessMath.util.misc_helper import MiscHelper
from AccessMath.data.meta_data_DB import MetaDataDB

from AccessMath.speaker.actions.foreground_estimator import ForegroundEstimator
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
    output_bboxes_dir = output_dir + "/" + config.get("SPEAKER_ACTION_CLASSIFICATION_BBOXES_DIR")
    video_metadata_dir = output_dir + "/" + config.get_str("SPEAKER_ACTION_VIDEO_META_DATA_DIR")
    fg_mask_dir = output_dir + "/" + config.get_str("SPEAKER_FG_ESTIMATION_MASK_DIR")
    os.makedirs(fg_mask_dir, exist_ok=True)

    dataset_name = config.get("SPEAKER_TESTING_SET_NAME")
    testing_set = database.datasets[dataset_name]

    speaker_exp_factor = config.get_float("SPEAKER_FG_ESTIMATION_SPK_EXPANSION_FACTOR")
    min_mask_frames = config.get_int("SPEAKER_FG_ESTIMATION_MIN_MASK_FRAMES")
    mask_exp_radius = config.get_int("SPEAKER_FG_ESTIMATION_MASK_EXPANSION_RADIUS")

    if len(sys.argv) >= 3:
        use_ground_truth = int(sys.argv[2]) > 0
    else:
        use_ground_truth = False

    for current_lecture in testing_set:
        bbox_filename = output_bboxes_dir + "/" + database.name + "_" + current_lecture.title + ".csv"
        frame_idxs, actions, body_bboxes, rh_bboxes = ResultReader.read_bbox_file(bbox_filename, use_ground_truth)

        info_filename = video_metadata_dir + "/" + database.name + "_" + current_lecture.title + ".pickle"
        video_info = MiscHelper.dump_load(info_filename)

        fg_estimator = ForegroundEstimator(video_info["width"], video_info["height"],
                                           speaker_exp_factor, min_mask_frames, mask_exp_radius)

        fg_mask = fg_estimator.get_mask(frame_idxs, actions, body_bboxes, rh_bboxes)

        # cv2.imshow(current_lecture.id, fg_mask)
        # cv2.waitKey()

        flag, raw_data = cv2.imencode(".png", fg_mask)

        output_filename = fg_mask_dir + "/" + database.name + "_" + current_lecture.title + ".pickle"
        MiscHelper.dump_save(raw_data, output_filename)


if __name__ == "__main__":
    main()
