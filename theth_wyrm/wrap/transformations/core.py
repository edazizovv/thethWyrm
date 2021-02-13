#


#
import numpy
import pandas


#


#
class Transformation:

    def _fit(self, X, y):
        raise Exception("Not realized")

    def _predict(self, X):
        raise Exception("Not realized")

    def fit(self, X, y):
        return self._fit(X=X, y=y)

    def predict(self, X):
        return self._predict(X=X)


class TransScikit(Transformation):

    def __init__(self, _model, *args, **kwargs):

        self._model = _model
        self.model = _model(*args, **kwargs)

    def _fit(self, X, y):

        if len(y.shape) == 1:
            y_ = y.reshape(-1, 1)
        else:
            y_ = y.copy()

        Z = numpy.concatenate([X, y_], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y_[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))

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
        predicted = self.model.transform(X_)
        Z = numpy.full(shape=(X.shape[0], predicted.shape[1]), fill_value=numpy.nan, dtype=numpy.float64)
        Z[nan_mask, :] = predicted
        return Z
