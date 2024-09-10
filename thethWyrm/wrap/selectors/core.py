#


#


#


#
class Selector:

    @property
    def support(self):
        raise Exception("Not realized")

    def _fit(self, X, y):
        raise Exception("Not realized")

    def _predict(self, X):
        raise Exception("Not realized")

    def fit(self, X, y):
        self._fit(X, y)

    def predict(self, X):
        return self._predict(X)
