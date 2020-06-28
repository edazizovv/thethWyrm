#
import json
import numpy
import pandas
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression as sk_OLR
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor as sk_KNR
from sklearn.tree import DecisionTreeRegressor as sk_DTR
from sklearn.ensemble import ExtraTreesRegressor as sk_ETR, RandomForestRegressor as sk_RFR
from sklearn.svm import SVR as sk_SVR
from lightgbm import LGBMRegressor as li_LBR
from xgboost import XGBRegressor as xg_XBR


class AnyModel:

    def __init__(self, data, name, model, params_current, params_space):

        self.data = data

        self.name = name
        self.model = model
        self.model_ = None
        self.params_current = params_current
        if params_space is None:
            self.params_space = params_space
        elif isinstance(params_space, str):
            with open('./models_params.json') as f:
                models_params = json.load(f)
            self.params_space = models_params[self.name]
        elif isinstance(params_space, dict):
            self.params_space = params_space
        else:
            raise Exception("what is yours params space?")

    def fit(self):
        raise Exception("Not yet!")

    def plot_fit(self):
        raise Exception("Not yet!")

    def melt_coeff(self):
        raise Exception("Not yet!")

    def melt_significance(self):
        raise Exception("Not yet!")

    def melt_correlation(self):
        raise Exception("Not yet!")

    def correlation_matrix(self):
        raise Exception("Not yet!")

    def storm(self):
        raise Exception("Not yet!")

    def stabilize(self):
        raise Exception("Not yet!")

    def hyper_opt(self):
        raise Exception("Not yet!")

    def summarize(self):
        raise Exception("Not yet!")

    def predict(self, dim0_mask):
        raise Exception("Not yet!")


class rfe_KNR(sk_KNR):

    def fit(self, X, Y):
        params = self.get_params()
        model = sk_KNR(**params)
        self.rfe = RFECV(model)
        self.rfe.fit(X, Y)

    def predict(self, X):
        return self.rfe.predict(X)

    def score(self, X, Y):
        return self.rfe.score(X, Y)


class rfe_DTR(sk_DTR):

    def fit(self, X, Y):
        params = self.get_params()
        model = sk_DTR(**params)
        self.rfe = RFECV(model)
        self.rfe.fit(X, Y)

    def predict(self, X):
        return self.rfe.predict(X)

    def score(self, X, Y):
        return self.rfe.score(X, Y)


class rfe_ETR(sk_ETR):

    def fit(self, X, Y):
        params = self.get_params()
        model = sk_ETR(**params)
        self.rfe = RFECV(model)
        self.rfe.fit(X, Y)

    def predict(self, X):
        return self.rfe.predict(X)

    def score(self, X, Y):
        return self.rfe.score(X, Y)


class rfe_RFR(sk_RFR):

    def fit(self, X, Y):
        params = self.get_params()
        model = sk_RFR(**params)
        self.rfe = RFECV(model)
        self.rfe.fit(X, Y)

    def predict(self, X):
        return self.rfe.predict(X)

    def score(self, X, Y):
        return self.rfe.score(X, Y)


class rfe_LBR(li_LBR):

    def fit(self, X, Y):
        params = self.get_params()
        model = li_LBR(**params)
        self.rfe = RFECV(model)
        self.rfe.fit(X, Y)

    def predict(self, X):
        return self.rfe.predict(X)

    def score(self, X, Y):
        return self.rfe.score(X, Y)


class rfe_XBR(xg_XBR):

    def fit(self, X, Y):
        params = self.get_params()
        model = xg_XBR(**params)
        self.rfe = RFECV(model)
        self.rfe.fit(X, Y)

    def predict(self, X):
        return self.rfe.predict(X)

    def score(self, X, Y):
        return self.rfe.score(X, Y)


class rfe_OLR(sk_OLR):

    def fit(self, X, Y):
        params = self.get_params()
        model = sk_OLR(**params)
        self.rfe = RFECV(model)
        self.rfe.fit(X, Y)

    def predict(self, X):
        return self.rfe.predict(X)

    def score(self, X, Y):
        return self.rfe.score(X, Y)


class rfe_SVR(sk_SVR):

    def fit(self, X, Y):
        params = self.get_params()
        model = sk_SVR(**params)
        self.rfe = RFECV(model)
        self.rfe.fit(X, Y)

    def predict(self, X):
        return self.rfe.predict(X)

    def score(self, X, Y):
        return self.rfe.score(X, Y)


class ScikitLearnAPIModel(AnyModel):

    def __init__(self, data, name, model, params_current, params_space):
        super().__init__(data=data, name=name, model=model, params_current=params_current, params_space=params_space)

    def fit(self):

        X_train, Y_train = self.data.values

        self.model_ = self.model(**self.params_current)
        self.model_.fit(X_train, Y_train.ravel())

    def _correlator(self, y, x):

        model_ = self.model(**self.params_current)
        model_.fit(x.reshape(-1, 1), y)
        y_hat = model_.predict(x.reshape(-1, 1))
        score = r2_score(y, y_hat)
        return score

    def correlation_matrix(self):

        X, _ = self.data.values
        names, _ = self.data.names
        table = pandas.DataFrame(data=X, columns=names)
        corr_table = table.corr(method=self._correlator)
        return corr_table

    def predict(self, dim0_mask=None):
        if dim0_mask is None:
            X, _ = self.data.values
        else:
            old_d0 = self.data.mask.d1
            self.data.mask.d0 = dim0_mask
            X, _ = self.data.values
            self.data.mask.d0 = old_d0
        predicted = self.model_.predict(X)
        return predicted

    def melt_coeff(self, min_left=1, step=1, cv=5):

        X, Y = self.data.values
        model_ = self.model(**self.params_current)
        rfe = RFECV(model_, min_features_to_select=min_left, step=step, cv=cv)
        rfe.fit(X, Y.ravel())
        self.data.mask.d1 = rfe.support_
        X, Y = self.data.values
        self.model_ = self.model(**self.params_current)
        self.model_.fit(X, Y)


class OLR(ScikitLearnAPIModel):

    def __init__(self, data):
        params_current = {'fit_intercept': False, 'n_jobs': -1}
        super().__init__(data=data, name='OLR', model=sk_OLR, params_current=params_current, params_space=None)

    def _summarize(self):

        # https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression

        X, Y = self.data.values
        Y = Y.ravel()
        names, _ = self.data.names

        params = numpy.append(self.model_.intercept_, self.model_.coef_)
        predictions = self.model_.predict(X)

        newX = pandas.DataFrame({"Constant": numpy.ones(len(X))}).join(pandas.DataFrame(X))
        MSE = (sum((Y - predictions) ** 2)) / (len(newX) - len(newX.columns))

        # Note if you don't want to use a DataFrame replace the two lines above with
        # newX = np.append(np.ones((len(X),1)), X, axis=1)
        # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

        var_b = MSE * (numpy.linalg.inv(numpy.dot(newX.T, newX)).diagonal())
        sd_b = numpy.sqrt(var_b)
        ts_b = params / sd_b

        p_values = [2 * (1 - stats.t.cdf(numpy.abs(i), (len(newX) - 1))) for i in ts_b]

        sd_b = numpy.round(sd_b, 3)
        ts_b = numpy.round(ts_b, 3)
        p_values = numpy.round(p_values, 3)
        params = numpy.round(params, 4)

        myDF3 = pandas.DataFrame()
        myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["Probabilities"] = [params, sd_b,
                                                                                                      ts_b,
                                                                                                      p_values]

        myDF3.index = numpy.array(['intercept'] + [x for x in names])
        return myDF3

    def summarize(self):

        return self._summarize()

    def _melt_significance_censor(self, significance):

        p_values = self._summarize()['Probabilities'].values[1:]
        diff = p_values < significance
        self.data.mask.d1[self.data.mask.d1] = diff
        return diff

    def melt_significance(self, significance=0.05):

        done = False
        while not done:

            XX, YY = self.data.values
            self.model_ = self.model(**self.params_current)
            self.model_.fit(XX, YY.ravel())

            diff = self._melt_significance_censor(significance)
            mask = self.data.mask.d1
            done = mask.sum() == 0 or diff.all() == 1

    def hyper_opt(self, melt=None):
        # add meltors!

        if self.params_space is None:
            raise Exception("your params_space is empty, idk what to search")
        else:
            if melt is None:
                model_ = self.model()
                params_space = self.params_space
            elif melt == 'coeff':
                model_ = rfe_OLR()
                params_space = self.params_space
            else:
                raise Exception("Not yet!")
            X, Y = self.data.values
            gscv = GridSearchCV(model_, params_space)
            gscv.fit(X, Y.ravel())
            self.params_current = gscv.best_params_


class KNR(ScikitLearnAPIModel):

    def __init__(self, data, params_space):
        params_current = {}
        super().__init__(data=data, name='KNR', model=sk_KNR, params_current=params_current, params_space=params_space)

    def melt_coeff(self, min_left=1, step=1, cv=5):
        raise Exception("Not yet!")

    def hyper_opt(self, melt=None):
        # add meltors!

        if self.params_space is None:
            raise Exception("your params_space is empty, idk what to search")
        else:
            if melt is None:
                model_ = self.model()
                params_space = self.params_space
            elif melt == 'coeff':
                model_ = rfe_KNR()
                params_space = self.params_space
            else:
                raise Exception("Not yet!")
            X, Y = self.data.values
            gscv = GridSearchCV(model_, params_space)
            gscv.fit(X, Y.ravel())
            self.params_current = gscv.best_params_


class DTR(ScikitLearnAPIModel):

    def __init__(self, data, params_space):
        params_current = {}
        super().__init__(data=data, name='DTR', model=sk_DTR, params_current=params_current, params_space=params_space)

    def melt_coeff(self, min_left=1, step=1, cv=5):
        raise Exception("Not yet!")

    def plot_fit(self):
        # here should be a pic of the tree
        raise Exception("Not yet!")

    def hyper_opt(self, melt=None):
        # add meltors!

        if self.params_space is None:
            raise Exception("your params_space is empty, idk what to search")
        else:
            if melt is None:
                model_ = self.model()
                params_space = self.params_space
            elif melt == 'coeff':
                model_ = rfe_DTR()
                params_space = self.params_space
            else:
                raise Exception("Not yet!")
            X, Y = self.data.values
            gscv = GridSearchCV(model_, params_space)
            gscv.fit(X, Y.ravel())
            self.params_current = gscv.best_params_


class ETR(ScikitLearnAPIModel):

    def __init__(self, data, params_space):
        params_current = {}
        super().__init__(data=data, name='ETR', model=sk_ETR, params_current=params_current, params_space=params_space)

    def hyper_opt(self, melt=None):
        # add meltors!

        if self.params_space is None:
            raise Exception("your params_space is empty, idk what to search")
        else:
            if melt is None:
                model_ = self.model()
                params_space = self.params_space
            elif melt == 'coeff':
                model_ = rfe_ETR()
                params_space = self.params_space
            else:
                raise Exception("Not yet!")
            X, Y = self.data.values
            gscv = GridSearchCV(model_, params_space)
            gscv.fit(X, Y.ravel())
            self.params_current = gscv.best_params_


class RFR(ScikitLearnAPIModel):

    def __init__(self, data, params_space):
        params_current = {}
        super().__init__(data=data, name='RFR', model=sk_RFR, params_current=params_current, params_space=params_space)

    def hyper_opt(self, melt=None):
        # add meltors!

        if self.params_space is None:
            raise Exception("your params_space is empty, idk what to search")
        else:
            if melt is None:
                model_ = self.model()
                params_space = self.params_space
            elif melt == 'coeff':
                model_ = rfe_RFR()
                params_space = self.params_space
            else:
                raise Exception("Not yet!")
            X, Y = self.data.values
            gscv = GridSearchCV(model_, params_space)
            gscv.fit(X, Y.ravel())
            self.params_current = gscv.best_params_


class LBR(ScikitLearnAPIModel):

    def __init__(self, data, params_space):
        params_current = {}
        super().__init__(data=data, name='LBR', model=li_LBR, params_current=params_current, params_space=params_space)

    def hyper_opt(self, melt=None):
        # add meltors!

        if self.params_space is None:
            raise Exception("your params_space is empty, idk what to search")
        else:
            if melt is None:
                model_ = self.model()
                params_space = self.params_space
            elif melt == 'coeff':
                model_ = rfe_LBR()
                params_space = self.params_space
            else:
                raise Exception("Not yet!")
            X, Y = self.data.values
            gscv = GridSearchCV(model_, params_space)
            gscv.fit(X, Y.ravel())
            self.params_current = gscv.best_params_


class XBR(ScikitLearnAPIModel):

    def __init__(self, data, params_space):
        params_current = {}
        super().__init__(data=data, name='XBR', model=xg_XBR, params_current=params_current, params_space=params_space)

    def hyper_opt(self, melt=None):
        # add meltors!

        if self.params_space is None:
            raise Exception("your params_space is empty, idk what to search")
        else:
            if melt is None:
                model_ = self.model()
                params_space = self.params_space
            elif melt == 'coeff':
                model_ = rfe_XBR()
                params_space = self.params_space
            else:
                raise Exception("Not yet!")
            X, Y = self.data.values
            gscv = GridSearchCV(model_, params_space)
            gscv.fit(X, Y.ravel())
            self.params_current = gscv.best_params_


class SVR(ScikitLearnAPIModel):
    # note: in future it shall select which model to use: linearsvr, svr, or nusvr depending on parameters passed to it
    # but currently we simply use svr

    def __init__(self, data, params_space):
        params_current = {}
        super().__init__(data=data, name='SVR', model=sk_SVR, params_current=params_current, params_space=params_space)
    # also coef_ status should be checked:
    # """
    # This is only available in the case of a linear kernel
    # """
    # (from the official docs)

    def hyper_opt(self, melt=None):
        # add meltors!

        if self.params_space is None:
            raise Exception("your params_space is empty, idk what to search")
        else:
            if melt is None:
                model_ = self.model()
                params_space = self.params_space
            elif melt == 'coeff':
                model_ = rfe_SVR()
                params_space = self.params_space
            else:
                raise Exception("Not yet!")
            X, Y = self.data.values
            gscv = GridSearchCV(model_, params_space)
            gscv.fit(X, Y.ravel())
            self.params_current = gscv.best_params_

