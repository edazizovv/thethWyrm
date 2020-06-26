#
import numpy


class DataMask:

    def __init__(self, d0, d1):
        self.d0, self.d1 = numpy.array(d0), numpy.array(d1)

    @property
    def mask(self):
        return numpy.ix_(self.d0, self.d1)


class DataHandler:

    def __init__(self, data_frame, target, qualitative, quantitative):

        self.data_frame = data_frame
        self.target = target
        self.qualitative = qualitative
        self.quantitative = quantitative
        for category in self.qualitative:
            self.data_frame[category] = self.data_frame[category].astype('category')
        for numeric in self.quantitative:
            self.data_frame[numeric] = self.data_frame[numeric].astype('float64')
        self._mask = None
        self.set_default_mask()

    def set_default_mask(self):
        d0 = [True] * self.data_frame.shape[0]
        d1 = [x != self.target for x in self.data_frame.columns.values]
        d0, d1 = numpy.array(d0), numpy.array(d1)
        self._mask = DataMask(d0=d0, d1=d1)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        if mask is None:
            self.set_default_mask()
        elif len(mask) == 2:
            if mask[0] is None:
                mask[0] = [True] * self.data_frame.shape[0]
            if mask[1] is None:
                mask[1] = numpy.array([True] * self.data_frame.shape[1])
            d0, d1 = numpy.array(mask[0]), numpy.array(mask[1])
            self._mask = DataMask(d0=d0, d1=d1)
        else:
            raise Exception("what is your mask? idk")

    @property
    def values(self):
        return self.data_frame.values[self.mask.mask], self.data_frame[[self.target]].values

    @property
    def names(self):
        return self.data_frame.columns.values[self.mask.d1], numpy.array([self.target])
