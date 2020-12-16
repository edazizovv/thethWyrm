#
import numpy
import pandas
from scipy import stats

from sklearn.linear_model import LinearRegression as sk_OLS
from statsmodels.api import GLS as sm_GLS


"""
General class has the following structure:

MYMODEL:

    > __init__(with model parameters)
    > fit
    > fit_plot
    > predict
    > summarize

"""


class OLS:

    def __init__(self):
        self.model = sk_OLS

    def fit(self, X, Y):
        self.model = self.model(n_jobs=-1, fit_intercept=False)
        self.model.fit(X, Y)

    def fit_plot(self):
        raise NotImplemented("Not yet!")

    def predict(self, X):
        return self.model.predict(X)

    def summarize(self, X, Y, names):
        # https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression

        params = numpy.append(self.model.intercept_, self.model.coef_)
        predictions = self.model.predict(X)

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


class GLS:

    def __init__(self, sigma):
        self.sigma = sigma
        self.model = sm_GLS

    def fit(self, X, Y):
        self.model = self.model(Y, X, self.sigma)
        self.model = self.model.fit()

    def fit_plot(self):
        raise NotImplemented("Not yet!")

    def predict(self, X):
        return self.model.get_prediction(X)

    def summarize(self, X):
        raise NotImplemented("Not yet!")


class FGLS:

    def __init__(self):
        self.sup_model = sk_OLS
        self.model = sm_GLS

    def fit(self, X, Y):
        raise NotImplemented("Not yet!")

    def fit_plot(self):
        raise NotImplemented("Not yet!")

    def predict(self, X):
        return self.model.get_prediction(X)

    def summarize(self, X):
        raise NotImplemented("Not yet!")
