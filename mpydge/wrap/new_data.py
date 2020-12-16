#
import numpy
import pandas


#


#
class DataPrimitive:

    def __init__(self, data_frame, qualitative, quantitative):
        self.data_frame = data_frame

        self._qualitative = qualitative
        self._quantitative = quantitative

    @property
    def values(self):
        return pandas.DataFrame(self.data_frame)

    @property
    def qualitative(self):
        return pandas.DataFrame(self.data_frame.loc[:, self._qualitative])

    @property
    def quantitative(self):
        return pandas.DataFrame(self.data_frame.loc[:, self._quantitative])


class MaskHandler:

    def __init__(self, train, validation, test):
        self.train = train
        self.validation = validation
        self.test = test


class MaskyConfig:

    def __init__(self, data_len, sweet_couple_rate, bucket_rate, step):
        self.data_len = data_len
        self.sweet_couple_rate = sweet_couple_rate
        self.bucket_rate = bucket_rate
        self.step = step

        self._current_step = 0
        self._identified = 0
        self._bucket_size = int(self.data_len * self.bucket_rate)
        self._train_size = int(self._bucket_size * self.sweet_couple_rate)
        self._test_size = self._bucket_size - self._train_size

    def tick(self):
        train_mask = numpy.array([self._current_step <= x <= self._current_step + self._train_size for x in range(self.data_len)])
        test_mask = numpy.array([self._current_step + self._train_size <= x <= self._current_step + self._bucket_size for x in range(self.data_len)])
        if self.step + self._bucket_size >= self.data_len:
            self._identified += 1
            self.step = 0
        else:
            self.step += self._bucket_size
        mask = MaskHandler(train=train_mask, validation=None, test=test_mask)
        return mask, self._identified


class DataHandler:

    def __init__(self, data_frame, qualitative, quantitative, sample_mask=None):

        self.masky = None
        self._masky_id = 0

        self.data_frame = data_frame
        self.sample_mask = sample_mask
        self.qualitative = qualitative
        self.quantitative = quantitative

        for category in self.qualitative:
            self.data_frame[category] = self.data_frame[category].astype('category')
        for numeric in self.quantitative:
            self.data_frame[numeric] = self.data_frame[numeric].astype('float64')

        self.train = None
        self.validation = None
        self.test = None

    def copy(self):
        copy = DataHandler(data_frame=self.data_frame.copy(),
                           qualitative=self.qualitative, quantitative=self.quantitative, sample_mask=self.sample_mask)
        copy.sample()
        return copy

    def update_factors(self, new_factors):
        for factor_name in new_factors.keys():
            if factor_name in self.factors:
                if (factor_name in self.qualitative and new_factors[factor_name] == 'category') or \
                        (factor_name in self.quantitative and new_factors[factor_name] == 'float64'):
                    pass
                elif factor_name in self.qualitative and new_factors[factor_name] == 'category':
                    self.qualitative.append(factor_name)
                    self.quantitative.remove(factor_name)
                elif factor_name in self.quantitative and new_factors[factor_name] == 'float64':
                    self.quantitative.append(factor_name)
                    self.qualitative.remove(factor_name)
                else:
                    raise Exception("Unacceptable factor type")
            else:
                print(factor_name in self.data_frame.columns.values)
                print(self.data_frame.columns.values)
                self.data_frame[factor_name] = self.data_frame[factor_name].astype(new_factors[factor_name])
                if new_factors[factor_name] == 'category':
                    self.qualitative.append(factor_name)
                elif new_factors[factor_name] == 'float64':
                    self.quantitative.append(factor_name)
                else:
                    raise Exception("Unacceptable factor type")
        # self.sample()

    def sample(self):
        if self.sample_mask is not None:
            if self.sample_mask.train is not None:
                self.train = self._result(self.sample_mask.train)
            else:
                print('train sample occurs to be None')
            if self.sample_mask.validation is not None:
                self.validation = self._result(self.sample_mask.validation)
            else:
                print('validation sample occurs to be None')
            if self.sample_mask.test is not None:
                self.test = self._result(self.sample_mask.test)
            else:
                print('test sample occurs to be None')
        else:
            raise Exception("Not realized yet")

    @property
    def factors(self):
        return self.qualitative + self.quantitative

    """
    @property
    def train(self):
        return self._result(self.sample_mask.train)

    @property
    def validation(self):
        return self._result(self.sample_mask.validation)

    @property
    def test(self):
        return self._result(self.sample_mask.test)
    """
    def _result(self, mask):
        return pandas.DataFrame(self.data_frame.loc[mask, :])

    def masky_set(self, sweet_couple_rate, bucket_rate, step):
        self.masky = MaskyConfig(self.data_frame.shape[0], sweet_couple_rate, bucket_rate, step)

    def masky_tick(self):
        self.sample_mask, _id = self.masky.tick()
        if self._masky_id == _id:
            self.sample()
        return _id
