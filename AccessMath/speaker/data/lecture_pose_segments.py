
import numpy as np
import pandas as pd

from AccessMath.speaker.actions.pose_feature_extractor import PoseFeatureExtractor

class LecturePoseSegments:
    def __init__(self, norm_factor, point_dim):
        self.segments = []
        self.norm_factor = norm_factor
        self.point_dim = point_dim

    def get_all_labels(self):
        return [segment.label for segment in self.segments]

    def get_all_segment_frames(self):
        return [(segment.frame_start, segment.frame_end) for segment in self.segments]

    @staticmethod
    def InitializeFromLectureFile(filename, normalization_bone, remove_confidence):
        pose_data_csv = pd.read_csv(filename)

        # 'initialize' the openpose data of current lecture
        # get lecture_pose_data with/without removing the confidency column
        filtered_pose_data, point_dim = LecturePoseSegments.get_openpose_data(pose_data_csv, remove_confidence)

        # correct the aspect ratio problem of videos
        # lec_pose_current_new[:, 0::2] *= train_ratio
        norm_joint_1, norm_joint_2 = normalization_bone

        # get normalization bone dist from all frames
        pair_dist, __ = PoseFeatureExtractor.point_pair_distances(norm_joint_1, norm_joint_2, filtered_pose_data,
                                                                  point_dim)

        # calculate the global average of bone dist and use it as norm factor
        norm_factor, __, __ = PoseFeatureExtractor.time_seg_avg_dist(pair_dist, pair_dist.shape[0])

        # add norm_factor in to data_dict
        # start with empty object ....
        lecture_segments = LecturePoseSegments(norm_factor[0], point_dim)

        return lecture_segments, filtered_pose_data

    @staticmethod
    def get_openpose_data(lecture_csv_data, remove_confidence):
        # get points data only from openpose csv file without the frame index columns

        openpose_data = lecture_csv_data.values
        if remove_confidence:
            # remove the frame index and confidence columns
            # remove confidence and frame index columns
            rm_columns = [i * 3 for i in range(int((openpose_data.shape[1] - 1) / 3 + 1))]

            openpose_data = np.delete(openpose_data, rm_columns, 1)
            p_dim = 2
            print("Lecture pose data shape: {0}, with point dim of {1}".format(openpose_data.shape, p_dim))

        else:
            # if keep the confidency value, then only remove the frame index column
            openpose_data = np.delete(openpose_data, [0], 1)
            p_dim = 3
            print("Lecture pose data shape: {0}, with point dim of {1}".format(openpose_data.shape, p_dim))

        return openpose_data, p_dim