#
import numpy
import pandas


#


#
class Fold:

    def __init__(self, data_frame, mask):

        # self._data_frame = data_frame.copy(deep=False)
        self._data_frame = data_frame
        self._mask = mask

    def __getitem__(self, item):
        return self._data_frame.loc[self._mask, item]

    def __setitem__(self, key, value):
        self._data_frame.loc[self._mask, key] = value

    def __repr__(self):
        return repr(self._data_frame.loc[self._mask, :])

    def __getattr__(self, item):
        if item in ['_data_frame', '_mask']:
            r = getattr(self, item)
        else:
            r = getattr(self._data_frame, item)
        return r

    """
    def __setattr__(self, key, value):
        if key in ['_data_frame', '_mask']:
            self.__dict__[key] = value
        else:
            print(2)
            # self._data_frame[key] = value
            # self._data_frame.__setattr__(key, value)
            # setattr(self._data_frame, key, value)
            self._data_frame[key] = value
        # self._data_frame.__setattr__(key, value)

        # try:
        #     setattr(self, key, value)
        # except AttributeError:
        #     try:
        #         setattr(self._data_frame, key, value)
        #     except AttributeError as e:
        #         raise e
    """


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

        self._sample_mask_train = None
        self._sample_mask_test = None

        self.train = None
        self.test = None

    def copy(self):
        copy = DataHandler(data_frame=self.data_frame.copy(),
                           qualitative=self.qualitative, quantitative=self.quantitative,
                           sample_points=self.sample_points)
        copy._current_sample_point_ix = self._current_sample_point_ix
        copy.sample()
        return copy

    def sample(self):
        self._current_sample_point_ix += 1

        if len(self.sample_points) == 1:
            self._sample_mask_train = [x <= self.sample_rate(0) for x in range(self.data_frame.shape[0])]
            self._sample_mask_test = [self.sample_rate(0) < x for x in range(self.data_frame.shape[0])]
        else:
            if self._current_sample_point_ix == 0:
                self._sample_mask_train = [x <= self.sample_rate(0) for x in range(self.data_frame.shape[0])]
                self._sample_mask_test = [self.sample_rate(0) < x <= self.sample_rate(1) for x in
                                          range(self.data_frame.shape[0])]
            elif self._current_sample_point_ix == (len(self.sample_points) - 1):
                self._sample_mask_train = [self.sample_rate(self._current_sample_point_ix - 1) < x <= self.sample_rate(
                    self._current_sample_point_ix) for x in range(self.data_frame.shape[0])]
                self._sample_mask_test = [self.sample_rate(self._current_sample_point_ix) < x for x in
                                          range(self.data_frame.shape[0])]
            else:
                self._sample_mask_train = [self.sample_rate(self._current_sample_point_ix - 1) < x <= self.sample_rate(
                    self._current_sample_point_ix) for x in range(self.data_frame.shape[0])]
                self._sample_mask_test = [self.sample_rate(self._current_sample_point_ix) < x <= self.sample_rate(
                    self._current_sample_point_ix + 1) for x in range(self.data_frame.shape[0])]

        self.train = Fold(data_frame=self.data_frame, mask=self._sample_mask_train)
        self.test = Fold(data_frame=self.data_frame, mask=self._sample_mask_test)

    """
    @property
    def train(self):
        print('ho')
        # return self._result(self._sample_mask_train)
        return self.data_frame[self._sample_mask_train]

    @property
    def test(self):
        # return self._result(self._sample_mask_test)
        return self.data_frame[self._sample_mask_test]
    
    @train.setter
    def train(self, value):
        print('hi')
        self.data_frame.loc[self._sample_mask_train, :] = value

    @test.setter
    def test(self, value):
        self.data_frame.loc[self._sample_mask_test, :] = value
    """

    def sample_rate(self, i):
        return int(self.data_frame.shape[0] * self.sample_points[i])

    # def _result(self, mask):
    #     return self.data_frame.loc[mask, :]
