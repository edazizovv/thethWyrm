#
import numpy
import pandas
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR as SVR_
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.decomposition import PCA as PCA_, TruncatedSVD, DictionaryLearning, FastICA, NMF as NMF_
from sklearn.decomposition import LatentDirichletAllocation, KernelPCA, SparsePCA, MiniBatchSparsePCA
from sklearn.manifold import MDS as MDS_, TSNE as TSNE_
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from umap import UMAP as UMAP_

import torch
#
from m_utils.measures import r2_adj
from m_utils.transformations import LogPctTransformer, Whitener, HypeTan  # , Axe  <-- coming soon
from neuro_new import WrappedNumericOnlyGene
from neuro_supernova import WrappedNumericOnlyGene as WrappedNumericOnlyGeneSupernova


#
def MAE(y_true, y_pred):
    Z = numpy.concatenate([y_true.reshape(-1, 1), y_pred.reshape(-1, 1)], axis=1)
    Z = numpy.array(Z, dtype=numpy.float32)
    Z[Z == numpy.inf] = numpy.nan
    Z[Z == -numpy.inf] = numpy.nan
    nan_mask = ~pandas.isna(Z).any(axis=1)
    y_true_, y_pred_ = y_true[nan_mask], y_pred[nan_mask]
    if y_true_.shape[0] == 0:
        return numpy.nan
    else:
        if y_true_.shape[0] != y_true.shape[0]:
            print('MAE: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(y_true.shape[0] - y_true_.shape[0]))
        return mean_absolute_error(y_true=y_true_, y_pred=y_pred_)


def R2(y_true, y_pred):
    Z = numpy.concatenate([y_true.reshape(-1, 1), y_pred.reshape(-1, 1)], axis=1)
    Z = numpy.array(Z, dtype=numpy.float32)
    Z[Z == numpy.inf] = numpy.nan
    Z[Z == -numpy.inf] = numpy.nan
    nan_mask = ~pandas.isna(Z).any(axis=1)
    y_true_, y_pred_ = y_true[nan_mask], y_pred[nan_mask]
    if y_true_.shape[0] == 0:
        return numpy.nan
    else:
        if y_true_.shape[0] != y_true.shape[0]:
            print('MAE: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(y_true.shape[0] - y_true_.shape[0]))
        return r2_score(y_true=y_true_, y_pred=y_pred_)


def R2_adj(y_true, y_pred, dim1):
    Z = numpy.concatenate([y_true.reshape(-1, 1), y_pred.reshape(-1, 1)], axis=1)
    Z = numpy.array(Z, dtype=numpy.float32)
    Z[Z == numpy.inf] = numpy.nan
    Z[Z == -numpy.inf] = numpy.nan
    nan_mask = ~pandas.isna(Z).any(axis=1)
    y_true_, y_pred_ = y_true[nan_mask], y_pred[nan_mask]
    if y_true_.shape[0] == 0:
        return numpy.nan
    else:
        if y_true_.shape[0] != y_true.shape[0]:
            print('MAE: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(y_true.shape[0] - y_true_.shape[0]))
        return r2_adj(y_true=y_true_, y_pred=y_pred_, dim0=Z.shape[0], dim1=dim1)


class OLR:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = LinearRegression(*args, **kwargs)

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            self.rfe = RFECV(self.model)
            self.rfe.fit(X_, y_)
        else:
            self.model.fit(X_, y_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        Z = numpy.full(shape=(X.shape[0], 1), fill_value=numpy.nan, dtype=numpy.float64)
        if self.rfe_cv:
            Z[nan_mask, :] = self.rfe.predict(X_).reshape(-1, 1)
        else:
            Z[nan_mask, :] = self.model.predict(X_).reshape(-1, 1)
        return Z


class KNR:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = KNeighborsRegressor(*args, **kwargs)

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            self.rfe = RFECV(self.model)
            self.rfe.fit(X_, y_)
        else:
            self.model.fit(X_, y_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        Z = numpy.full(shape=(X.shape[0], 1), fill_value=numpy.nan, dtype=numpy.float64)
        if self.rfe_cv:
            Z[nan_mask, :] = self.rfe.predict(X_).reshape(-1, 1)
        else:
            Z[nan_mask, :] = self.model.predict(X_).reshape(-1, 1)
        return Z


class DTR:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = DecisionTreeRegressor(*args, **kwargs)

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            self.rfe = RFECV(self.model)
            self.rfe.fit(X_, y_)
        else:
            self.model.fit(X_, y_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        Z = numpy.full(shape=(X.shape[0], 1), fill_value=numpy.nan, dtype=numpy.float64)
        if self.rfe_cv:
            Z[nan_mask, :] = self.rfe.predict(X_).reshape(-1, 1)
        else:
            Z[nan_mask, :] = self.model.predict(X_).reshape(-1, 1)
        return Z


class ETR:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = ExtraTreesRegressor(*args, **kwargs)

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            self.rfe = RFECV(self.model)
            self.rfe.fit(X_, y_)
        else:
            self.model.fit(X_, y_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        Z = numpy.full(shape=(X.shape[0], 1), fill_value=numpy.nan, dtype=numpy.float64)
        if self.rfe_cv:
            Z[nan_mask, :] = self.rfe.predict(X_).reshape(-1, 1)
        else:
            Z[nan_mask, :] = self.model.predict(X_).reshape(-1, 1)
        return Z


class RFR:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = RandomForestRegressor(*args, **kwargs)

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            self.rfe = RFECV(self.model)
            self.rfe.fit(X_, y_)
        else:
            self.model.fit(X_, y_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        Z = numpy.full(shape=(X.shape[0], 1), fill_value=numpy.nan, dtype=numpy.float64)
        if self.rfe_cv:
            Z[nan_mask, :] = self.rfe.predict(X_).reshape(-1, 1)
        else:
            Z[nan_mask, :] = self.model.predict(X_).reshape(-1, 1)
        return Z

    def set_params(self, **kwargs):
        self.model.set_params(**kwargs)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class SVR:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = SVR_(*args, **kwargs)

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            self.rfe = RFECV(self.model)
            self.rfe.fit(X_, y_)
        else:
            self.model.fit(X_, y_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        Z = numpy.full(shape=(X.shape[0], 1), fill_value=numpy.nan, dtype=numpy.float64)
        if self.rfe_cv:
            Z[nan_mask, :] = self.rfe.predict(X_).reshape(-1, 1)
        else:
            Z[nan_mask, :] = self.model.predict(X_).reshape(-1, 1)
        return Z


class GBR:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = GradientBoostingRegressor(*args, **kwargs)

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            self.rfe = RFECV(self.model)
            self.rfe.fit(X_, y_)
        else:
            self.model.fit(X_, y_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        Z = numpy.full(shape=(X.shape[0], 1), fill_value=numpy.nan, dtype=numpy.float64)
        if self.rfe_cv:
            Z[nan_mask, :] = self.rfe.predict(X_).reshape(-1, 1)
        else:
            Z[nan_mask, :] = self.model.predict(X_).reshape(-1, 1)
        return Z


class GBR:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = GradientBoostingRegressor(*args, **kwargs)

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            self.rfe = RFECV(self.model)
            self.rfe.fit(X_, y_)
        else:
            self.model.fit(X_, y_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        Z = numpy.full(shape=(X.shape[0], 1), fill_value=numpy.nan, dtype=numpy.float64)
        if self.rfe_cv:
            Z[nan_mask, :] = self.rfe.predict(X_).reshape(-1, 1)
        else:
            Z[nan_mask, :] = self.model.predict(X_).reshape(-1, 1)
        return Z


class LBR:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = LGBMRegressor(*args, **kwargs)

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            self.rfe = RFECV(self.model)
            self.rfe.fit(X_, y_)
        else:
            self.model.fit(X_, y_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        Z = numpy.full(shape=(X.shape[0], 1), fill_value=numpy.nan, dtype=numpy.float64)
        if self.rfe_cv:
            Z[nan_mask, :] = self.rfe.predict(X_).reshape(-1, 1)
        else:
            Z[nan_mask, :] = self.model.predict(X_).reshape(-1, 1)
        return Z


class XBR:

    def __init__(self, rfe_cv, grid_cv=None, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.grid = None
        self.grid_cv = grid_cv
        self.model = XGBRegressor(*args, **kwargs)

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))

        if self.grid_cv is not None:
            self.grid = GridSearchCV(estimator=self.model, param_grid=self.grid_cv)
            self.grid.fit(X_, y_)
            self.model = XGBRegressor(**self.grid.best_params_)
            if self.rfe_cv:
                self.rfe = RFECV(self.model)
                self.rfe.fit(X_, y_)
        elif self.rfe_cv:
            self.rfe = RFECV(self.model)
            self.rfe.fit(X_, y_)
        else:
            self.model.fit(X_, y_)
        """
        if self.rfe_cv:
            self.rfe = RFECV(self.model)
            self.rfe.fit(X_, y_)
        elif self.grid_cv is not None:
            self.grid = GridSearchCV(estimator=self.model, param_grid=self.grid_cv)
            self.grid.fit(X_, y_)
        else:
            self.model.fit(X_, y_)
        """

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        Z = numpy.full(shape=(X.shape[0], 1), fill_value=numpy.nan, dtype=numpy.float64)
        if self.rfe_cv:
            Z[nan_mask, :] = self.rfe.predict(X_).reshape(-1, 1)
        elif self.grid_cv is not None:
            Z[nan_mask, :] = self.grid.predict(X_).reshape(-1, 1)
        else:
            Z[nan_mask, :] = self.model.predict(X_).reshape(-1, 1)
        return Z


class Insane:

    def __init__(self, my_name):
        self.my_name = my_name
        self.store = None

    def say_my_name(self):
        return self.my_name

    def fit(self, array):

        if self.my_name == 'Nothing':

            pass

        elif self.my_name == 'LnPct':

            trf = LogPctTransformer()
            trf.fit(array)
            self.store = trf

        elif self.my_name == 'TanhLnPct':

            trf0 = LogPctTransformer()
            trf0.fit(array)
            array_ = trf0.transform(array)
            trf1 = HypeTan()
            trf1.fit(array_)
            trf = [trf0, trf1]
            self.store = trf

        elif self.my_name == 'Whiten':

            trf = Whitener()
            trf.fit(array)
            self.store = trf

        elif self.my_name == 'TanhWhiten':

            trf0 = Whitener()
            trf0.fit(array)
            array_ = trf0.transform(array)
            trf1 = HypeTan()
            trf1.fit(array_)
            trf = [trf0, trf1]
            self.store = trf

        elif self.my_name == 'AxeLnPct':

            """
            trf0 = LogPctTransformer()
            trf0.fit(array)
            array_ = trf0.tranform(array)
            trf1 = Axe()
            trf1.fit(array_)
            """

            raise Exception("Axe is not ready!")

        elif self.my_name == 'AxeWOELnPct':

            raise Exception("Axe is not ready!")

        else:

            raise Exception("Not Yet!")

    def forward(self, array):

        if self.my_name == 'Nothing':

            return array

        elif self.my_name == 'LnPct':

            return self.store.transform(array)

        elif self.my_name == 'TanhLnPct':

            return self.store[1].transform(self.store[0].transform(array))

        elif self.my_name == 'Whiten':

            return self.store.transform(array)

        elif self.my_name == 'TanhWhiten':

            return self.store[1].transform(self.store[0].transform(array))

        elif self.my_name == 'AxeLnPct':

            # return self.store[1].transform(self.store[0].transform(array))

            raise Exception("It is coming soon...")

        elif self.my_name == 'AxeWOELnPct':

            raise Exception("It is coming soon...")

        else:

            raise Exception("Not Yet!")

    def backward(self, array):

        if self.my_name == 'Nothing':

            return array

        elif self.my_name == 'LnPct':

            return self.store.inverse_transform(array)

        elif self.my_name == 'TanhLnPct':

            return self.store[0].inverse_transform(self.store[1].inverse_transform(array))

        elif self.my_name == 'Whiten':

            return self.store.inverse_transform(array)

        elif self.my_name == 'TanhWhiten':

            return self.store[0].inverse_transform(self.store[1].inverse_transform(array))

        elif self.my_name == 'AxeLnPct':

            # return self.store[0].inverse_transform(self.store[1].inverse_transform(array))

            raise Exception("It is coming soon...")

        elif self.my_name == 'AxeWOELnPct':

            raise Exception("It is coming soon...")

        else:

            raise Exception("Not Yet!")


class Neakt:

    def __init__(self, masked, coded):
        self.masked = masked
        self.coded = coded
        self.transformers = list(self.masked.keys())
        self.masks = [self.masked[key] for key in self.transformers]
        self.n = len(self.transformers)

    def say_my_name(self):

        return self.coded

    def fit(self, X, Y):

        array = X.copy()

        for j in range(self.n):
            self.transformers[j].fit(array[:, self.masks[j]])

    def __predict(self, X):

        array_ = []

        for j in range(self.n):
            array_.append(self.transformers[j].forward(X[:, self.masks[j]]))

        array_ = numpy.concatenate(array_, axis=1)
        return array_

    def predict(self, X):

        array_ = X.copy()

        for j in range(self.n):
            array_[:, self.masks[j]] = self.transformers[j].forward(array_[:, self.masks[j]])  # .reshape(-1, 1)

        return array_

    def backward(self, array):

        array_ = array.copy()

        for j in range(self.n):
            # array_ = self.transformers[-j - 1].inverse_transform(array_[:, self.masks[-j - 1]])
            # print('iter {0}'.format(j))
            # print('full array')
            # print(array_)
            # print('changeable part')
            # print(array_[:, self.masks[-j - 1]])
            rr = self.transformers[-j - 1].backward(array_[:, self.masks[-j - 1]])  # .reshape(-1, 1)
            if len(rr.shape) == 1:
                array_[:, self.masks[-j - 1]] = rr.reshape(-1, 1)
            elif len(rr.shape) == 2:
                array_[:, self.masks[-j - 1]] = rr
            else:
                raise Exception("Something went wrong, check with debug pls")
        # print('final')
        # print(array_)
        return array_


class SimpleNumericNN:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = WrappedNumericOnlyGeneSupernova(*args, **kwargs)

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)                                                                 # !
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))

        X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.3)
        y_train, y_val = y_train.reshape(-1, 1), y_val.reshape(-1, 1)

        X_train_, X_val_ = torch.tensor(X_train, dtype=torch.float), torch.tensor(X_val, dtype=torch.float)
        y_train_, y_val_ = torch.tensor(y_train, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)

        self.model.fit(X_train=X_train_, y_train=y_train_, X_val=X_val_, y_val=y_val_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        Z = numpy.full(shape=(X.shape[0], 1), fill_value=numpy.nan, dtype=numpy.float64)
        X_ = torch.tensor(X[nan_mask, :], dtype=torch.float)
        Z[nan_mask, :] = self.model.predict(X_).reshape(-1, 1)

        return Z


class NeuralEmbedder:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = WrappedNumericOnlyGeneSupernova(*args, **kwargs)

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)                                                                 # !
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))

        X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.3)
        y_train, y_val = y_train.reshape(-1, 1), y_val.reshape(-1, 1)

        X_train_, X_val_ = torch.tensor(X_train, dtype=torch.float), torch.tensor(X_val, dtype=torch.float)
        y_train_, y_val_ = torch.tensor(y_train, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)

        self.model.fit(X_train=X_train_, y_train=y_train_, X_val=X_val_, y_val=y_val_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))

        X_ = torch.tensor(X[nan_mask, :], dtype=torch.float)

        print(self.model.embed(X_, n=-1).shape)

        predicted = self.model.embed(X_, n=-1)
        Z = numpy.full(shape=(X.shape[0], predicted.shape[1]), fill_value=numpy.nan, dtype=numpy.float64)
        Z[nan_mask, :] = predicted

        return Z


class PCA:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = PCA_(*args, **kwargs)

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            self.model.fit(X_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            predicted = self.model.transform(X_)
            Z = numpy.full(shape=(X.shape[0], predicted.shape[1]), fill_value=numpy.nan, dtype=numpy.float64)
            Z[nan_mask, :] = predicted
        return Z


class TSVD:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = TruncatedSVD(*args, **kwargs)

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            self.model.fit(X_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            predicted = self.model.transform(X_)
            Z = numpy.full(shape=(X.shape[0], predicted.shape[1]), fill_value=numpy.nan, dtype=numpy.float64)
            Z[nan_mask, :] = predicted
        return Z


class DICL:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = DictionaryLearning(*args, **kwargs)

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            self.model.fit(X_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            predicted = self.model.transform(X_)
            Z = numpy.full(shape=(X.shape[0], predicted.shape[1]), fill_value=numpy.nan, dtype=numpy.float64)
            Z[nan_mask, :] = predicted
        return Z


class FICA:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = FastICA(*args, **kwargs)

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            self.model.fit(X_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            predicted = self.model.transform(X_)
            Z = numpy.full(shape=(X.shape[0], predicted.shape[1]), fill_value=numpy.nan, dtype=numpy.float64)
            Z[nan_mask, :] = predicted
        return Z


class NMF:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = NMF_(*args, **kwargs)

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            self.model.fit(X_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            predicted = self.model.transform(X_)
            Z = numpy.full(shape=(X.shape[0], predicted.shape[1]), fill_value=numpy.nan, dtype=numpy.float64)
            Z[nan_mask, :] = predicted
        return Z


class LDA:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = LatentDirichletAllocation(*args, **kwargs)

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            self.model.fit(X_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            predicted = self.model.transform(X_)
            Z = numpy.full(shape=(X.shape[0], predicted.shape[1]), fill_value=numpy.nan, dtype=numpy.float64)
            Z[nan_mask, :] = predicted
        return Z


class KPCA:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = KernelPCA(*args, **kwargs)

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            self.model.fit(X_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            predicted = self.model.transform(X_)
            Z = numpy.full(shape=(X.shape[0], predicted.shape[1]), fill_value=numpy.nan, dtype=numpy.float64)
            Z[nan_mask, :] = predicted
        return Z


class SPCA:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = SparsePCA(*args, **kwargs)

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            self.model.fit(X_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            predicted = self.model.transform(X_)
            Z = numpy.full(shape=(X.shape[0], predicted.shape[1]), fill_value=numpy.nan, dtype=numpy.float64)
            Z[nan_mask, :] = predicted
        return Z


class MBSPCA:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = MiniBatchSparsePCA(*args, **kwargs)

    def fit(self, X, y):
        Z = numpy.concatenate([X, y.reshape(-1, 1)], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        X_, y_ = X[~pandas.isna(Z).any(axis=1), :], y[~pandas.isna(Z).any(axis=1)]
        if Z.shape[0] != X.shape[0]:
            print('FIT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            self.model.fit(X_)

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            predicted = self.model.transform(X_)
            Z = numpy.full(shape=(X.shape[0], predicted.shape[1]), fill_value=numpy.nan, dtype=numpy.float64)
            Z[nan_mask, :] = predicted
        return Z


class MDS:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = MDS_(*args, **kwargs)

    def fit(self, X, y):
        pass

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            predicted = self.model.fit_transform(X_)
            Z = numpy.full(shape=(X.shape[0], predicted.shape[1]), fill_value=numpy.nan, dtype=numpy.float64)
            Z[nan_mask, :] = predicted
        return Z


class TSNE:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = TSNE_(*args, **kwargs)

    def fit(self, X, y):
        pass

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            predicted = self.model.fit_transform(X_)
            Z = numpy.full(shape=(X.shape[0], predicted.shape[1]), fill_value=numpy.nan, dtype=numpy.float64)
            Z[nan_mask, :] = predicted
        return Z


class UMAP:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = UMAP_(*args, **kwargs)

    def fit(self, X, y):
        pass

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            predicted = self.model.fit_transform(X_)
            Z = numpy.full(shape=(X.shape[0], predicted.shape[1]), fill_value=numpy.nan, dtype=numpy.float64)
            Z[nan_mask, :] = predicted
        return Z


class GRP:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = GaussianRandomProjection(*args, **kwargs)

    def fit(self, X, y):
        pass

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            predicted = self.model.fit_transform(X_)
            Z = numpy.full(shape=(X.shape[0], predicted.shape[1]), fill_value=numpy.nan, dtype=numpy.float64)
            Z[nan_mask, :] = predicted
        return Z


class SRP:

    def __init__(self, rfe_cv, *args, **kwargs):
        self.rfe = None
        self.rfe_cv = rfe_cv
        self.model = SparseRandomProjection(*args, **kwargs)

    def fit(self, X, y):
        pass

    def predict(self, X):
        Z = numpy.concatenate([X], axis=1)
        Z = numpy.array(Z, dtype=numpy.float32)
        Z[Z == numpy.inf] = numpy.nan
        Z[Z == -numpy.inf] = numpy.nan
        nan_mask = ~pandas.isna(Z).any(axis=1)
        X_ = X[nan_mask, :]
        if Z.shape[0] != X.shape[0]:
            print('PREDICT: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(X.shape[0] - X_.shape[0]))
        if self.rfe_cv:
            raise Exception("PCA could not be processed with RFE_CV")
        else:
            predicted = self.model.fit_transform(X_)
            Z = numpy.full(shape=(X.shape[0], predicted.shape[1]), fill_value=numpy.nan, dtype=numpy.float64)
            Z[nan_mask, :] = predicted
        return Z


class ZerosReductor:

    def __init__(self):
        self.support_ = None

    def fit(self, X, Y):
        self.support_ = numpy.array([~(X[:, j] == 0).all() for j in range(X.shape[1])])

    def predict(self, X):
        X_ = X[:, self.support_]
        return X_


class LuCienLaChance:

    def __init__(self):
        self.mean = numpy.nan

    def fit(self, X, Y):
        self.mean = X.mean()

    def predict(self, X):
        X_ = numpy.roll(X, shift=1)
        X_[0] = X[0]
        X_ = X_.reshape(-1, 1)
        return X_
