#
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


class DataHandler:

    def __init__(self, data_frame, qualitative, quantitative, sample_mask=None):

        self.data_frame = data_frame
        self.sample_mask = sample_mask
        self.qualitative = qualitative
        self.quantitative = quantitative

        for category in self.qualitative:
            self.data_frame[category] = self.data_frame[category].astype('category')
        for numeric in self.quantitative:
            self.data_frame[numeric] = self.data_frame[numeric].astype('float64')

        """
        self.train = None
        self.validation = None
        self.test = None
        """

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
                self.data_frame[factor_name] = self.data_frame[factor_name].astype(new_factors[factor_name])
                if new_factors[factor_name] == 'category':
                    self.qualitative.append(factor_name)
                elif new_factors[factor_name] == 'float64':
                    self.quantitative.append(factor_name)
                else:
                    raise Exception("Unacceptable factor type")

    def sample(self):
        raise Exception("Not realized yet")

    @property
    def factors(self):
        return self.qualitative + self.quantitative

    @property
    def train(self):
        return self._result(self.sample_mask.train)

    @property
    def validation(self):
        return self._result(self.sample_mask.validation)

    @property
    def test(self):
        return self._result(self.sample_mask.test)

    def _result(self, mask):
        return pandas.DataFrame(self.data_frame.loc[:, mask])

    """
    def _result(self, mask):
        data_frame = pandas.DataFrame(self.data_frame.loc[:, mask])
        return DataPrimitive(data_frame=data_frame,
                             qualitative=self.qualitative, quantitative=self.quantitative)
    """