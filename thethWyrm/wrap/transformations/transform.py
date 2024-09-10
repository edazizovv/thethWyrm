#


#
import numpy
import pandas
from sklearn.decomposition import PCA as PCA_, TruncatedSVD, DictionaryLearning, FastICA, NMF as NMF_
from sklearn.decomposition import LatentDirichletAllocation, KernelPCA, SparsePCA, MiniBatchSparsePCA
from sklearn.manifold import MDS as MDS_, TSNE as TSNE_
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from umap import UMAP as UMAP_


#
from theth_wyrm.wrap.transformations.core import Transformation, TransScikit


#
class PC(Transformation):

    def __init__(self, rate=False):
        self.rate = rate

    def _fit(self, X, y):
        pass

    def _predict(self, X):

        X_ = pandas.DataFrame(X).T
        X_ = X_.pct_change().dropna().T.values
        if not self.rate:
            X_ = X_ + 1

        return X_


class CP(Transformation):

    def __init__(self, rate=False):
        self.rate = rate

    def _fit(self, X, y):

        if X.shape[1] != 2:
            raise Exception("X is expected to be n x 2 matrix")

    def _predict(self, X):

        if X.shape[1] != 2:
            raise Exception("X is expected to be n x 2 matrix")

        X_ = X.copy().T
        if not self.rate:
            X_[0, :] = X_[0, :] + 1
        X__ = X_[0, :] * X_[1, :]

        return X__


class LG(Transformation):

    def __init__(self, base='e', plus=0):
        self.base = base
        self.plus = plus

    def _fit(self, X, y):
        pass

    def _predict(self, X):

        if self.base == 'e':
            X_ = numpy.log(self.plus + X)
        else:
            X_ = numpy.log(self.plus + X, self.base)

        return X_


class EX(Transformation):

    def __init__(self, base='e', plus=0):
        self.base = base
        self.plus = plus

    def _fit(self, X, y):
        pass

    def _predict(self, X):

        if self.base == 'e':
            X_ = numpy.exp(X) - self.plus
        else:
            X_ = numpy.power(X, self.base) - self.plus

        return X_


class PCA(TransScikit):

    def __init__(self, *args, **kwargs):
        super().__init__(_model=PCA_, *args, **kwargs)


class KPCA(TransScikit):

    def __init__(self, *args, **kwargs):
        super().__init__(_model=KernelPCA, *args, **kwargs)


class SPCA(TransScikit):

    def __init__(self, *args, **kwargs):
        super().__init__(_model=SparsePCA, *args, **kwargs)


class MBSPCA(TransScikit):

    def __init__(self, *args, **kwargs):
        super().__init__(_model=MiniBatchSparsePCA, *args, **kwargs)


class LDA(TransScikit):

    def __init__(self, *args, **kwargs):
        super().__init__(_model=LatentDirichletAllocation, *args, **kwargs)


class TSVD(TransScikit):

    def __init__(self, *args, **kwargs):
        super().__init__(_model=TruncatedSVD, *args, **kwargs)


class DICL(TransScikit):

    def __init__(self, *args, **kwargs):
        super().__init__(_model=DictionaryLearning, *args, **kwargs)


class FICA(TransScikit):

    def __init__(self, *args, **kwargs):
        super().__init__(_model=FastICA, *args, **kwargs)


class NMF(TransScikit):

    def __init__(self, *args, **kwargs):
        super().__init__(_model=NMF_, *args, **kwargs)


class MDS(TransScikit):

    def __init__(self, *args, **kwargs):
        super().__init__(_model=MDS_, *args, **kwargs)


class TSNE(TransScikit):

    def __init__(self, *args, **kwargs):
        super().__init__(_model=TSNE_, *args, **kwargs)


class UMAP(TransScikit):

    def __init__(self, *args, **kwargs):
        super().__init__(_model=UMAP_, *args, **kwargs)


class GRP(TransScikit):

    def __init__(self, *args, **kwargs):
        super().__init__(_model=GaussianRandomProjection, *args, **kwargs)


class SRP(TransScikit):

    def __init__(self, *args, **kwargs):
        super().__init__(_model=SparseRandomProjection, *args, **kwargs)
