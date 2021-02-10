#
import numpy
import pandas
from boruta import BorutaPy
from sklearn.feature_selection import GenericUnivariateSelect, mutual_info_regression

#


#

class NoRazor:

    def __init__(self):
        self.supp_ = None
        pass

    @property
    def __bleed__(self):
        return False

    @property
    def support_(self):
        return self.supp_

    def fit(self, X, y):
        self.supp_ = numpy.ones(shape=(X.shape[1],), dtype=bool)

    def predict(self, X):
        return X

    
class MutualInfoRazor:

    def __init__(self, percentile=50):
        self.percentile = percentile
        self.transformer = GenericUnivariateSelect(score_func=mutual_info_regression,
                                                   mode='percentile', param=self.percentile)

    @property
    def support_(self):
        return self.transformer.get_support()

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        self.transformer.fit(X_, y_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        RZ = self.transformer.transform(X_)
        Z = numpy.full(shape=(X.shape[0], RZ.shape[1]), fill_value=numpy.nan)
        Z[nan_mask, :] = RZ
        return Z


class BorutaRazor:

    def __init__(self, model, model_kwargs, boruta_kwargs):

        self.model = model(**model_kwargs)
        self.boruta = BorutaPy(self.model, **boruta_kwargs)

    @property
    def support_(self):
        return self.boruta.support_

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        self.boruta.fit(X_, y_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        RZ = self.boruta.transform(X_)
        Z = numpy.full(shape=(X.shape[0], RZ.shape[1]), fill_value=numpy.nan)
        Z[nan_mask, :] = RZ
        return Z
