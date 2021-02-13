#


#
import numpy
import pandas


#
from theth_wyrm.wrap.transformations.core import Transformation


#
class PC(Transformation):

    def __init__(self, rate=False):
        self.rate = rate

    def _fit(self, X, y):
        pass

    def _predict(self, X):

        X_ = pandas.DataFrame(X).T
        X_ = X_.pct_change().dropna().T.values
        if not self.rate:
            X_ = X_ + 1

        return X_


class CP(Transformation):

    def __init__(self, rate=False):
        self.rate = rate

    def _fit(self, X, y):

        if X.shape[1] != 2:
            raise Exception("X is expected to be n x 2 matrix")

    def _predict(self, X):

        if X.shape[1] != 2:
            raise Exception("X is expected to be n x 2 matrix")

        X_ = X.copy().T
        if not self.rate:
            X_[0, :] = X_[0, :] + 1
        X__ = X_[0, :] * X_[1, :]

        return X__


class LG(Transformation):

    def __init__(self, base='e', plus=0):
        self.base = base
        self.plus = plus

    def _fit(self, X, y):
        pass

    def _predict(self, X):

        if self.base == 'e':
            X_ = numpy.log(self.plus + X)
        else:
            X_ = numpy.log(self.plus + X, self.base)

        return X_


class EX(Transformation):

    def __init__(self, base='e', plus=0):
        self.base = base
        self.plus = plus

    def _fit(self, X, y):
        pass

    def _predict(self, X):

        if self.base == 'e':
            X_ = numpy.exp(X) - self.plus
        else:
            X_ = numpy.power(X, self.base) - self.plus

        return X_
