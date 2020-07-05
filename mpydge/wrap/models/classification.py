#
import numpy
import pandas
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as sk_KNC
from sklearn.tree import DecisionTreeClassifier as sk_DTC
from sklearn.ensemble import ExtraTreesClassifier as sk_ETC, RandomForestClassifier as sk_RFC
from sklearn.svm import SVC as sk_SVC
from lightgbm import LGBMClassifier as li_LBC
from xgboost import XGBClassifier as xg_XBC


#
from mpydge.wrap.models.core import AnyModel


#


class ClassificationModel(AnyModel):

    def __init__(self, data, name, preprocessor, model, params_current, params_space):
        super().__init__(data=data, name=name, preprocessor=preprocessor, model=model,
                         params_current=params_current, params_space=params_space)


class rfe_KNC(sk_KNC):

    def fit(self, X, Y):
        params = self.get_params()
        model = sk_KNC(**params)
        self.rfe = RFECV(model)
        self.rfe.fit(X, Y)

    def predict(self, X):
        return self.rfe.predict(X)

    def score(self, X, Y):
        return self.rfe.score(X, Y)


class rfe_DTC(sk_DTC):

    def fit(self, X, Y):
        params = self.get_params()
        model = sk_DTC(**params)
        self.rfe = RFECV(model)
        self.rfe.fit(X, Y)

    def predict(self, X):
        return self.rfe.predict(X)

    def score(self, X, Y):
        return self.rfe.score(X, Y)


class rfe_ETC(sk_ETC):

    def fit(self, X, Y):
        params = self.get_params()
        model = sk_ETC(**params)
        self.rfe = RFECV(model)
        self.rfe.fit(X, Y)

    def predict(self, X):
        return self.rfe.predict(X)

    def score(self, X, Y):
        return self.rfe.score(X, Y)


class rfe_RFC(sk_RFC):

    def fit(self, X, Y):
        params = self.get_params()
        model = sk_RFC(**params)
        self.rfe = RFECV(model)
        self.rfe.fit(X, Y)

    def predict(self, X):
        return self.rfe.predict(X)

    def score(self, X, Y):
        return self.rfe.score(X, Y)


class rfe_LBC(li_LBC):

    def fit(self, X, Y):
        params = self.get_params()
        model = li_LBC(**params)
        self.rfe = RFECV(model)
        self.rfe.fit(X, Y)

    def predict(self, X):
        return self.rfe.predict(X)

    def score(self, X, Y):
        return self.rfe.score(X, Y)


class rfe_XBC(xg_XBC):

    def fit(self, X, Y):
        params = self.get_params()
        model = xg_XBC(**params)
        self.rfe = RFECV(model)
        self.rfe.fit(X, Y)

    def predict(self, X):
        return self.rfe.predict(X)

    def score(self, X, Y):
        return self.rfe.score(X, Y)


class rfe_SVC(sk_SVC):

    def fit(self, X, Y):
        params = self.get_params()
        model = sk_SVC(**params)
        self.rfe = RFECV(model)
        self.rfe.fit(X, Y)

    def predict(self, X):
        return self.rfe.predict(X)

    def score(self, X, Y):
        return self.rfe.score(X, Y)


class ScikitLearnClassificationAPIModel(ClassificationModel):

    def __init__(self, data, name, preprocessor, model, params_current, params_space):
        super().__init__(data=data, name=name, preprocessor=preprocessor, model=model,
                         params_current=params_current, params_space=params_space)

    def fit(self):

        X_train, Y_train = self.data.values

        train = numpy.concatenate((X_train, Y_train), axis=1)

        self.preprocessor_ = self.preprocessor()
        self.preprocessor_.fit(train)
        train_tf = self.preprocessor_.transform(train)
        X_train_tf, Y_train_tf = train_tf[:, :-1], train_tf[:, -1]

        self.model_ = self.model(**self.params_current)
        self.model_.fit(X_train_tf, Y_train_tf.ravel())

    def _correlator(self, y, x):

        model_ = self.model(**self.params_current)
        model_.fit(x.reshape(-1, 1), y)
        y_hat = model_.predict(x.reshape(-1, 1))
        raise Exception("Not yet!")
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
            full = numpy.concatenate((X, _), axis=1)
            full_tf = self.preprocessor_.transform(full)
            X_tf, _tf = full_tf[:, :-1], full_tf[:, -1]
        else:
            old_d0 = self.data.mask.d1
            self.data.mask.d0 = dim0_mask
            X, _ = self.data.values
            full = numpy.concatenate((X, _), axis=1)
            full_tf = self.preprocessor_.transform(full)
            X_tf, _tf = full_tf[:, :-1], full_tf[:, -1]
            self.data.mask.d0 = old_d0
        predicted_tf = self.model_.predict(X_tf).reshape(-1, 1)
        predicted_full_tf = numpy.concatenate((X_tf, predicted_tf), axis=1)
        predicted_full = self.preprocessor_.inverse_transform(predicted_full_tf)
        _, predicted = predicted_full[:, :-1], predicted_full[:, -1]
        return predicted.ravel()

    def melt_coeff(self, min_left=1, step=1, cv=5):
        X, Y = self.data.values
        full = numpy.concatenate((X, Y), axis=1)
        full_tf = self.preprocessor_.transform(full)
        X_tf, Y_tf = full_tf[:, :-1], full_tf[:, -1]
        model_ = self.model(**self.params_current)
        rfe = RFECV(model_, min_features_to_select=min_left, step=step, cv=cv)
        rfe.fit(X_tf, Y_tf.ravel())
        self.data.mask.d1 = rfe.support_
        X, Y = self.data.values
        full = numpy.concatenate((X, Y), axis=1)
        full_tf = self.preprocessor_.transform(full)
        X_tf, Y_tf = full_tf[:, :-1], full_tf[:, -1]
        self.model_ = self.model(**self.params_current)
        self.model_.fit(X_tf, Y_tf)


class KNC(ScikitLearnClassificationAPIModel):

    def __init__(self, data, preprocessor, params_space):
        params_current = {}
        super().__init__(data=data, name='KNC', preprocessor=preprocessor, model=sk_KNC, params_current=params_current, params_space=params_space)

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
                model_ = rfe_KNC()
                params_space = self.params_space
            else:
                raise Exception("Not yet!")
            X, Y = self.data.values
            full = numpy.concatenate((X, Y), axis=1)
            full_tf = self.preprocessor_.transform(full)
            X_tf, Y_tf = full_tf[:, :-1], full_tf[:, -1]
            gscv = GridSearchCV(model_, params_space)
            gscv.fit(X_tf, Y_tf.ravel())
            self.params_current = gscv.best_params_


class DTC(ScikitLearnClassificationAPIModel):

    def __init__(self, data, preprocessor, params_space):
        params_current = {}
        super().__init__(data=data, name='DTC', preprocessor=preprocessor, model=sk_DTC,
                         params_current=params_current, params_space=params_space)

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
                model_ = rfe_DTC()
                params_space = self.params_space
            else:
                raise Exception("Not yet!")
            X, Y = self.data.values
            full = numpy.concatenate((X, Y), axis=1)
            full_tf = self.preprocessor_.transform(full)
            X_tf, Y_tf = full_tf[:, :-1], full_tf[:, -1]
            gscv = GridSearchCV(model_, params_space)
            gscv.fit(X_tf, Y_tf.ravel())
            self.params_current = gscv.best_params_


class ETC(ScikitLearnClassificationAPIModel):

    def __init__(self, data, preprocessor, params_space):
        params_current = {}
        super().__init__(data=data, name='ETC', preprocessor=preprocessor, model=sk_ETC,
                         params_current=params_current, params_space=params_space)

    def hyper_opt(self, melt=None):
        # add meltors!

        if self.params_space is None:
            raise Exception("your params_space is empty, idk what to search")
        else:
            if melt is None:
                model_ = self.model()
                params_space = self.params_space
            elif melt == 'coeff':
                model_ = rfe_ETC()
                params_space = self.params_space
            else:
                raise Exception("Not yet!")
            X, Y = self.data.values
            full = numpy.concatenate((X, Y), axis=1)
            full_tf = self.preprocessor_.transform(full)
            X_tf, Y_tf = full_tf[:, :-1], full_tf[:, -1]
            gscv = GridSearchCV(model_, params_space)
            gscv.fit(X_tf, Y_tf.ravel())
            self.params_current = gscv.best_params_


class RFC(ScikitLearnClassificationAPIModel):

    def __init__(self, data, preprocessor, params_space):
        params_current = {}
        super().__init__(data=data, name='RFC', preprocessor=preprocessor, model=sk_RFC,
                         params_current=params_current, params_space=params_space)

    def hyper_opt(self, melt=None):
        # add meltors!

        if self.params_space is None:
            raise Exception("your params_space is empty, idk what to search")
        else:
            if melt is None:
                model_ = self.model()
                params_space = self.params_space
            elif melt == 'coeff':
                model_ = rfe_RFC()
                params_space = self.params_space
            else:
                raise Exception("Not yet!")
            X, Y = self.data.values
            full = numpy.concatenate((X, Y), axis=1)
            full_tf = self.preprocessor_.transform(full)
            X_tf, Y_tf = full_tf[:, :-1], full_tf[:, -1]
            gscv = GridSearchCV(model_, params_space)
            gscv.fit(X_tf, Y_tf.ravel())
            self.params_current = gscv.best_params_


class LBC(ScikitLearnClassificationAPIModel):

    def __init__(self, data, preprocessor, params_space):
        params_current = {}
        super().__init__(data=data, name='LBC', preprocessor=preprocessor, model=li_LBC,
                         params_current=params_current, params_space=params_space)

    def hyper_opt(self, melt=None):
        # add meltors!

        if self.params_space is None:
            raise Exception("your params_space is empty, idk what to search")
        else:
            if melt is None:
                model_ = self.model()
                params_space = self.params_space
            elif melt == 'coeff':
                model_ = rfe_LBC()
                params_space = self.params_space
            else:
                raise Exception("Not yet!")
            X, Y = self.data.values
            full = numpy.concatenate((X, Y), axis=1)
            full_tf = self.preprocessor_.transform(full)
            X_tf, Y_tf = full_tf[:, :-1], full_tf[:, -1]
            gscv = GridSearchCV(model_, params_space)
            gscv.fit(X_tf, Y_tf.ravel())
            self.params_current = gscv.best_params_


class XBC(ScikitLearnClassificationAPIModel):

    def __init__(self, data, preprocessor, params_space):
        params_current = {}
        super().__init__(data=data, name='XBC', preprocessor=preprocessor, model=xg_XBC,
                         params_current=params_current, params_space=params_space)

    def hyper_opt(self, melt=None):
        # add meltors!

        if self.params_space is None:
            raise Exception("your params_space is empty, idk what to search")
        else:
            if melt is None:
                model_ = self.model()
                params_space = self.params_space
            elif melt == 'coeff':
                model_ = rfe_XBC()
                params_space = self.params_space
            else:
                raise Exception("Not yet!")
            X, Y = self.data.values
            full = numpy.concatenate((X, Y), axis=1)
            full_tf = self.preprocessor_.transform(full)
            X_tf, Y_tf = full_tf[:, :-1], full_tf[:, -1]
            gscv = GridSearchCV(model_, params_space)
            gscv.fit(X_tf, Y_tf.ravel())
            self.params_current = gscv.best_params_


class SVC(ScikitLearnClassificationAPIModel):
    # note: in future it shall select which model to use: linearsvr, svr, or nusvr depending on parameters passed to it
    # but currently we simply use svr

    def __init__(self, data, preprocessor, params_space):
        params_current = {}
        super().__init__(data=data, name='SVC', preprocessor=preprocessor, model=sk_SVC,
                         params_current=params_current, params_space=params_space)
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
                model_ = rfe_SVC()
                params_space = self.params_space
            else:
                raise Exception("Not yet!")
            X, Y = self.data.values
            full = numpy.concatenate((X, Y), axis=1)
            full_tf = self.preprocessor_.transform(full)
            X_tf, Y_tf = full_tf[:, :-1], full_tf[:, -1]
            gscv = GridSearchCV(model_, params_space)
            gscv.fit(X_tf, Y_tf.ravel())
            self.params_current = gscv.best_params_

