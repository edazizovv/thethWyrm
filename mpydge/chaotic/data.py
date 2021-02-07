#
import numpy
import pandas


#


#
class DataHandler:

    def __init__(self, data_frame, qualitative, quantitative, sample_points):

        self.data_frame = data_frame
        self.sample_points = sample_points
        self._current_sample_point_ix = -1
        self.qualitative = qualitative
        self.quantitative = quantitative

        for category in self.qualitative:
            self.data_frame[category] = self.data_frame[category].astype('category')
        for numeric in self.quantitative:
            self.data_frame[numeric] = self.data_frame[numeric].astype('float64')

        self.train = None
        self.test = None

    def copy(self):
        copy = DataHandler(data_frame=self.data_frame.copy(),
                           qualitative=self.qualitative, quantitative=self.quantitative, sample_points=self.sample_points)
        copy._current_sample_point_ix = self._current_sample_point_ix
        copy.sample()
        return copy

    def sample(self):
        self._current_sample_point_ix += 1

        if len(self.sample_points) == 1:
            sample_mask_train = [x <= self.sample_rate(0) for x in range(self.data_frame.shape[0])]
            sample_mask_test = [self.sample_rate(0) < x for x in range(self.data_frame.shape[0])]
        else:
            if self._current_sample_point_ix == 0:
                sample_mask_train = [x <= self.sample_rate(0) for x in range(self.data_frame.shape[0])]
                sample_mask_test = [self.sample_rate(0) < x <= self.sample_rate(1) for x in range(self.data_frame.shape[0])]
            elif self._current_sample_point_ix == (len(self.sample_points) - 1):
                sample_mask_train = [self.sample_rate(self._current_sample_point_ix - 1) < x <= self.sample_rate(self._current_sample_point_ix) for x in range(self.data_frame.shape[0])]
                sample_mask_test = [self.sample_rate(self._current_sample_point_ix) < x for x in range(self.data_frame.shape[0])]
            else:
                sample_mask_train = [self.sample_rate(self._current_sample_point_ix - 1) < x <= self.sample_rate(self._current_sample_point_ix) for x in range(self.data_frame.shape[0])]
                sample_mask_test = [self.sample_rate(self._current_sample_point_ix) < x <= self.sample_rate(self._current_sample_point_ix + 1) for x in range(self.data_frame.shape[0])]

        self.train = self._result(sample_mask_train)
        self.test = self._result(sample_mask_test)

    def sample_rate(self, i):
        return int(self.data_frame.shape[0] * self.sample_points[i])

    def _result(self, mask):
        return pandas.DataFrame(self.data_frame.loc[mask, :])
