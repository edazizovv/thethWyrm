#


#


#
from theth_wyrm.wrap.transformations.core import Transformation


#
class MassApply(Transformation):

    def __init__(self, masks_in, masks_out, transformations):
        self.masks_in = masks_in
        self.masks_out = masks_out
        self.transformations = transformations

    def _fit(self, X, y):

        for j in range(len(self.masks_in)):
            self.transformations[j].fit(X[:, self.masks_in[j]], y)

    def _predict(self, X):

        X_ = X.copy()
        for j in range(len(self.masks_in)):
            X_[:, self.masks_out[j]] = self.transformations[j].predict(X_[:, self.masks_in[j]])

        return X_
