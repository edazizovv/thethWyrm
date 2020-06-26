

class DataHandler:

    def __init__(self, data, target, qualitative, quantitative):

        self.data = data
        self.target = target
        self.qualitative = qualitative
        self.quantitative = quantitative
        for category in self.qualitative:
            self.data[category] = self.data[category].astype('category')
        for numeric in self.quantitative:
            self.data[numeric] = self.data[numeric].astype('float64')
        self._mask = [True] * self.data.shape[0]

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        if mask is None:
            mask = [x != self.data.target for x in self.data.data.columns.values]
        elif len(mask) == 2:
            if mask[0] is None:
                mask[0] = numpy.array([True] * self.data.data.shape[0])
            if mask[1] is None:
                mask[1] = numpy.array([True] * self.data.data.shape[1])
        else:
            raise Exception("what is your mask? idk")
        self._mask = mask

    @property
    def values(self):
        return self.data[self.mask[0], self.mask[1]].values, self.data[self.target].values

    @property
    def names(self):
        return self.data.columns.values[self.mask[1]], self.target
