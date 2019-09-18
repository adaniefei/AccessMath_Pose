
import numpy as np

class PoseSegmentData:
    def __init__(self, frame_start, frame_end, label, pose_data):
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.label = label
        self.pose_data = pose_data

    # get the bounding box of all points in a data segment
    @staticmethod
    def get_bbox_data(seg_data, dim):
        # using logic array!!!!
        data_stride = np.lib.stride_tricks.as_strided(seg_data, (int(seg_data.shape[1] / dim), seg_data.shape[0], dim),
                                                      (seg_data.dtype.itemsize * 2, seg_data.strides[0],
                                                       seg_data.strides[1]))

        data_reshape = data_stride.reshape((data_stride.shape[0] * data_stride.shape[1], data_stride.shape[2]))

        # check special case(when one point is not captured and valued as (0.0, 0.0))
        data_sum = data_reshape.sum(axis=1)
        valid_data = data_reshape[data_sum >= 0.1]

        if len(valid_data) < 2:
            x_min = -1
            x_max = -1
            y_min = -1
            y_max = -1
            return [x_min, x_max, y_min, y_max]

        x_min = valid_data[:, 0].min()
        x_max = valid_data[:, 0].max()
        y_min = valid_data[:, 1].min()
        y_max = valid_data[:, 1].max()

        return [x_min, x_max, y_min, y_max]

    # get the bounding box of all points in a data segment per frame
    @staticmethod
    def get_bbox_frame_data(seg_data, dim):
        frame_bbox = []
        for ind in range(0, seg_data.shape[0]):
            temp = []
            temp += PoseSegmentData.get_bbox_data(seg_data[None, ind], dim)  # trick!!!
            frame_bbox.append(temp)

        return np.array(frame_bbox)
