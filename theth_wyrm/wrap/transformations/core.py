#


#


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
