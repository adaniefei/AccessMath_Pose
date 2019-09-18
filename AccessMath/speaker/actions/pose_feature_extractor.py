
import numpy as np

from AccessMath.speaker.util.statistics import Statistics

class PoseFeatureExtractor:

    def __init__(self, feature_points, segment_length):
        self.feature_points = feature_points
        self.segment_length = segment_length

    def extract(self, lecture_pose_segments):

        features = []
        for feature_paramaters in self.feature_points:
            feature_target = feature_paramaters[0]

            if isinstance(feature_target, tuple):
                print("Add features from bone of {0}".format(feature_target))
                new_features = self.get_pair_feature(lecture_pose_segments, feature_paramaters)
            elif isinstance(feature_target, int):
                print("Add features from point of {0}".format(feature_target))
                new_features = self.get_point_feature(lecture_pose_segments, feature_paramaters)
            elif feature_target == "out_skeleton":
                print("Add features from skeleton captured confidence {0}".format(feature_target))
                new_features = self.get_skeleton_out_feature(lecture_pose_segments)
            else:
                print("Invalid feature configuration")
                print(feature_paramaters)
                new_features = []

            features.append(new_features)

        features = np.hstack(features)

        return features

    def get_feature_dataset(self, lecture_pose_segments):
        dataset = {
            "features": self.extract(lecture_pose_segments),
            "labels": lecture_pose_segments.get_all_labels(),
            "frame_infos": np.array(lecture_pose_segments.get_all_segment_frames()),
        }

        return dataset

    # add pair of points features
    def get_pair_feature(self, lecture_pose_segments, feature_paramaters):
        joint_1, joint_2 = feature_paramaters[0]
        abs_diff_flag = feature_paramaters[1]  # absolute difference on/raw difference on
        avg_flag = feature_paramaters[2]
        var_flag = feature_paramaters[3]
        med_flag = feature_paramaters[4]
        conf_flag = feature_paramaters[5]

        new_features = []

        for segment_data in lecture_pose_segments.segments:
            temp_feature = []

            # get dist between target points from all frames
            abs_all, diff_all = PoseFeatureExtractor.point_pair_distances(joint_1, joint_2, segment_data.pose_data,
                                                                          lecture_pose_segments.point_dim)

            # there is only (seg_len - 1) frame to frame difference
            abs_statics = Statistics.get_statics(abs_all, self.segment_length)
            diff_statics = Statistics.get_statics(diff_all, self.segment_length)

            if abs_diff_flag:
                if avg_flag:
                    temp_feature.append(abs_statics.average)
                if var_flag:
                    temp_feature.append(abs_statics.var)
                if med_flag:
                    temp_feature.append(abs_statics.median)
            else:
                if avg_flag:
                    temp_feature.append(diff_statics.average)
                if var_flag:
                    temp_feature.append(diff_statics.var)
                if med_flag:
                    temp_feature.append(diff_statics.median)

            temp_feature = np.hstack(temp_feature) / lecture_pose_segments.norm_factor
            temp_feature = list(temp_feature)

            if abs_diff_flag:
                if conf_flag:
                    temp_feature.append(abs_statics.confidence)
            else:
                if conf_flag:
                    temp_feature.append(diff_statics.confidence)

            new_features.append(temp_feature)

        a = np.array(new_features)

        print("size of feature from {0},{1} is: {2}".format(joint_1, joint_2, a.shape))

        return np.array(new_features)

    # add single point features
    def get_point_feature(self, lecture_pose_segments, feature_paramaters):
        test_target = feature_paramaters[0]
        # absolute difference statics on/off
        abs_avg_flag = feature_paramaters[1]
        abs_var_flag = feature_paramaters[2]
        abs_med_flag = feature_paramaters[3]
        # raw difference statics on/off
        diff_avg_flag = feature_paramaters[4]
        diff_var_flag = feature_paramaters[5]
        diff_med_flag = feature_paramaters[6]
        cov_matrix_flag = feature_paramaters[7]
        conf_flag = feature_paramaters[8]

        new_features = []

        for segment_data in lecture_pose_segments.segments:
            temp_feature = []

            # get test dist from all frames
            point_info = PoseFeatureExtractor.get_point_info(test_target, segment_data.pose_data,
                                                             lecture_pose_segments.point_dim)
            abs_f2f_diff, abs_f2f_diff_xy, f2f_diff_xy = point_info

            # there is only (seg_len - 1) frame to frame difference
            abs_statics = Statistics.get_statics(abs_f2f_diff_xy, self.segment_length - 1)
            diff_statics = Statistics.get_statics(f2f_diff_xy, self.segment_length - 1)

            if abs_avg_flag:
                temp_feature.append(abs_statics.average)
            if abs_var_flag:
                temp_feature.append(abs_statics.var)
            if abs_med_flag:
                temp_feature.append(abs_statics.median)
            if diff_avg_flag:
                temp_feature.append(diff_statics.average)
            if diff_var_flag:
                temp_feature.append(diff_statics.var)
            if diff_med_flag:
                temp_feature.append(diff_statics.median)
            # add cov_matrix info between point_x and point_y
            if cov_matrix_flag:
                f2f_diff_cov = Statistics.get_cov(f2f_diff_xy)
                temp_feature.append(f2f_diff_cov)

            temp_feature = np.hstack(temp_feature) / lecture_pose_segments.norm_factor
            temp_feature = list(temp_feature)

            if conf_flag:
                temp_feature.append(abs_statics.confidence)

            new_features.append(temp_feature)

        a = np.array(new_features)

        print("size of feature from {0} is {1}: ".format(test_target, a.shape))

        return np.array(new_features)

    # add skeleton out confidence feature
    def get_skeleton_out_feature(self, lecture_pose_segments):
        new_features = []

        for segment_data in lecture_pose_segments.segments:
            # we need to pick only one value of all points(x,y) since when skeleton is not captured,
            # every point's values is set to be "-1000"
            pick_info = segment_data.pose_data[:, 1]  # pick point_y since the point_x may be affected by the train_ratio value
            count = [val for val in pick_info if val == -1000]
            skeleton_conf = (self.segment_length - len(count)) / self.segment_length
            new_features.append([skeleton_conf])
        a = np.array(new_features)
        print("size of feature is {0}: ".format(a.shape))
        return np.array(new_features)

    # calculate pair-wise distance in all frames
    @staticmethod
    def point_pair_distances(joint_1, joint_2, points_data, dim):
        # get the point data for these joints....
        p0 = points_data[:, dim * joint_1:dim * joint_1 + dim]
        p1 = points_data[:, dim * joint_2:dim * joint_2 + dim]

        # calculate the pair-wise distance and difference
        diff = p0 - p1
        diff_square = diff ** 2
        diff_sum_col = np.sum(diff_square, axis=1)
        pair_dist = np.sqrt(diff_sum_col)

        # check the special case when point is not captured(0, 0, 0)
        sum_p0 = p0.sum(axis=1)
        sum_p1 = p1.sum(axis=1)
        min_sum = np.minimum(sum_p0, sum_p1)

        # keep the valid information only(without special cases)
        pair_dist = pair_dist[min_sum >= 0.1]
        diff = diff[min_sum >= 0.1]

        return pair_dist, diff

    # calculate point data in all frames
    @staticmethod
    def get_point_info(point_index, points_data, dim):
        p0 = points_data[:, dim * point_index:dim * point_index + dim]

        # check the special case when point is not captured(0, 0, 0)
        sum_p0 = p0.sum(axis=1)
        min_p0 = np.minimum(sum_p0[:-1], sum_p0[1:])
        f2f_diff = p0[:-1] - p0[1:]  # frame to frame difference

        # abs dist from point to point avg position
        diff_square = f2f_diff ** 2
        diff_sum_col = np.sum(diff_square, axis=1)
        abs_f2f_diff = np.sqrt(diff_sum_col)

        # keep the valid information only(without special cases)
        abs_f2f_diff = abs_f2f_diff[min_p0 >= 0.1]
        f2f_diff = f2f_diff[min_p0 >= 0.1]

        abs_f2f_diff_xy = abs(f2f_diff)

        return abs_f2f_diff, abs_f2f_diff_xy, f2f_diff

    # calculate the average dist during every 'seg_len' time sequence
    @staticmethod
    def time_seg_avg_dist(dist, seg_len, ignore_value=-10):
        frames = np.arange(dist.shape[0])

        avg_frames = []
        avg_dists = []
        var_dists = []
        for ind in range(0, dist.shape[0], seg_len):
            if ind + seg_len - 1 >= dist.shape[0]:
                break

            # middle position of each small sequencial segment
            avg_frames.append((frames[ind] + frames[ind + seg_len - 1]) / 2)
            segment_dist = dist[ind:ind + seg_len]

            # check special case when one point in the pair is not detected: (0, 0, 0)
            segment_dist = [item for item in segment_dist if item != ignore_value]
            if segment_dist == []:
                avg_dists.append(ignore_value)
                var_dists.append(ignore_value)
                continue

            avg_dist = np.mean(segment_dist)
            var_dist = np.var(segment_dist)
            avg_dists.append(avg_dist)
            var_dists.append(var_dist)

        return avg_dists, var_dists, avg_frames

    @staticmethod
    def combine_datasets(data_list, data_dictionary):
        data_x = []
        data_y = []
        frame_infos = []
        for vid in data_list:
            data_group = data_dictionary[vid.lower()]
            data_x += data_group["features"].tolist()
            data_y += data_group["labels"]
            frame_infos += data_group["frame_infos"].tolist()

        return data_x, data_y, frame_infos