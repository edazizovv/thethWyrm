#


#
import numpy


#
from theth_wyrm.wrap.models.core import SupM1D


#
class LuCienLaChance(SupM1D):

    def __init__(self):
        self.mean = numpy.nan

    def _fit(self, X, y):
        self.mean = X.mean()

    def _predict(self, X):
        X_ = numpy.roll(X, shift=1)
        X_[0] = X[0]
        X_ = X_.reshape(-1, 1)
        return X_
