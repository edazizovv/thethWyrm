

class AnyModel:

    def __init__(self, data, model, params_current, params_space):

        self.data = data

        self.model = model
        self.model_ = None
        self.params_current = params_current
        self.params_space = params_space

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


class OLS(AnyModel):

    def __init__(self, data):
        params_current = {'fit_intercept': False, 'n_jobs': -1}
        super().__init__(data=data, model=sk_OLS, params_current=params_current, params_space=None)

    def fit(self):

        X_train, Y_train = self.data.values

        self.model_ = self.model(**self.params_current)
        self.model_.fit(X_train, Y_train)

    def _summarize(self):

        # https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression

        X, Y = self.data.values
        names, _ = self.data.names

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

    def summarize(self):

        self._summarize()

    def _melt_coeff_censor(self, significance):

        p_values = self._summarize()['Probabilities'].values[1:]
        pv_mask = p_values < significance
        mask = self.data.mask
        mask[1] = pv_mask
        self.data.mask = mask

    def melt_significance(self, significance=0.05):

        done = False
        while not done:

            XX, YY = self.data.values
            model_ = self.model(**self.params_current)
            model_.fit(XX, YY)

            self._melt_coeff_censor(significance)
            mask = self.data.mask[1]
            done = mask.sum() == 0 or mask.all() == 1

    def predict(self, dim0_mask=None):
        if dim0_mask is None:
            X, _ = self.data.values
        else:
            old_mask = self.data.mask
            self.data.mask = [dim0_mask, None]
            X, _ = self.data.values
            self.data.mask = old_mask
        predicted = self.model.predict(X)
        return predicted
