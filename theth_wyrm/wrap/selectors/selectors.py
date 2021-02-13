#


#
import numpy


#
from theth_wyrm.wrap.selectors.core import Selector


#
class ZerosReductor(Selector):

    def __init__(self):
        self._support = None

    @property
    def support(self):
        return self._support

    def _fit(self, X, y):
        self._support = numpy.array([~(X[:, j] == 0).all() for j in range(X.shape[1])])

    def _predict(self, X):
        X_ = X[:, self.support]
        return X_


class NoRazor(Selector):

    def __init__(self):
        self._support = None
        pass

    @property
    def support(self):
        return self._support

    def _fit(self, X, y):
        self._support = numpy.ones(shape=(X.shape[1],), dtype=bool)

    def _predict(self, X):
        return X
