#


#
import numpy
import pandas
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV


#


#
class SupModel:

    def _fit(self, X, y):
        raise Exception("Not realized")

    def _predict(self, X):
        raise Exception("Not realized")

    def fit(self, X, y):
        return self._fit(X=X, y=y)

    def predict(self, X):
        return self._predict(X=X)


class SupM1D(SupModel):

    def fit(self, X, y):
        return self._fit(X=X, y=y.reshape(-1, 1))

    def predict(self, X):
        return self._predict(X=X).reshape(-1, 1)


class SupM1DScikit(SupM1D):

    def __init__(self, _model, rfe_enabled=False, grid_cv=None, *args, **kwargs):
        self.rfe = None
        self.rfe_enabled = rfe_enabled
        self.grid = None
        self.grid_cv = grid_cv
        self._model = _model
        self.model = self._model(*args, **kwargs)

    def _fit(self, X, y):

        Z = numpy.concatenate([X, y], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))

        if self.grid_cv is not None:
            self.grid = GridSearchCV(estimator=self.model, param_grid=self.grid_cv)
            self.grid.fit(X_, y_)
            self.model = self._model(**self.grid.best_params_)
            if self.rfe_enabled:
                self.rfe = RFECV(self.model)
                self.rfe.fit(X_, y_)
        elif self.rfe_enabled:
            self.rfe = RFECV(self.model)
            self.rfe.fit(X_, y_)
        else:
            self.model.fit(X_, y_)

    def _predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        Z = numpy.full(shape=(X.shape[0], 1), fill_value=numpy.nan, dtype=numpy.float64)
        if self.rfe_enabled:
            Z[nan_mask, :] = self.rfe.predict(X_).reshape(-1, 1)
        else:
            Z[nan_mask, :] = self.model.predict(X_).reshape(-1, 1)
        return Z
