#
import json
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import r2_score, mean_absolute_error

#
from m_utils.measures import r2_adj


#


class AnyModel:

    def __init__(self, data, name, preprocessor, model, params_current, params_space):

        self.data = data

        self.name = name
        self.preprocessor = preprocessor
        self.preprocessor_ = None
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

    def score(self, measure, dim0_mask=None, ts_report=False):
        Y_hat = self.predict(dim0_mask)
        if dim0_mask is None:
            X, Y = self.data.values
        else:
            old_d0 = self.data.mask.d0
            self.data.mask.d0 = dim0_mask
            X, Y = self.data.values
            self.data.mask.d0 = old_d0
        if measure == 'r2_adj':
            measured = r2_adj(Y, Y_hat, X.shape[0], X.shape[1])
        elif measure == 'mae':
            measured = mean_absolute_error(Y, Y_hat)
        elif measure == 'r2':
            measured = r2_score(Y, Y_hat)
        else:
            raise Exception("Not yet!")
        if ts_report:
            errors = Y.ravel() - Y_hat
            stationarity_score = adfuller(errors, regression='nc')[1]
            skewness = stats.skew(errors)
            return measured, stationarity_score, skewness
        else:
            return measured

    def ts_plot(self, train_dim0_mask=None, test_dim0_mask=None):
        raise Exception("Not yet!")

