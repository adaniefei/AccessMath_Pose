
import numpy as np

class Statistics:
    def __init__(self):
        self.average = [0.0]
        self.var = [0.0]
        self.median = [0.0]
        self.confidence = 0.0


    @staticmethod
    def get_statics(data_in, segment_len):
        data_out = Statistics()
        if len(data_in.shape) > 1:
            for i in range(1, len(data_in.shape)):
                data_out.average.append(0.0)
                data_out.var.append(0.0)
                data_out.median.append(0.0)

        if data_in.shape[0] == 0:
            return data_out

        data_out.average = np.mean(data_in, axis=0)
        data_out.var = np.var(data_in, axis=0)
        data_out.median = np.median(data_in, axis=0)
        data_out.confidence = data_in.shape[0]/segment_len

        return data_out

    @staticmethod
    def get_cov(points):
        cov_xy = 0.0

        if points.shape[0] <= 1:
            return cov_xy

        cov_matrix = np.cov(points.T)
        cov_xy = cov_matrix[0][1]
        return cov_xy
