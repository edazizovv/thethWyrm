#
import numpy


def cross_sectional(d0, test_rate=0.2):
    N = d0.shape[0]
    K = d0.sum()
    ai = numpy.array(numpy.arange(K))
    sampled = numpy.random.choice(ai, size=int(ai.shape[0] * test_rate), replace=False)
    train_ = ~numpy.isin(ai, sampled)
    test_ = numpy.isin(ai, sampled)
    train = numpy.zeros(shape=(N,), dtype=bool)
    test = numpy.zeros(shape=(N,), dtype=bool)
    train[d0] = train_
    test[d0] = test_
    return train, test


def time_serial(d0, n_folds=5, test_rate=0.2):
    N = d0.shape[0]
    K = d0.sum()
    ai = numpy.array(numpy.arange(K))
    thresh = int(ai.shape[0] * (1 - test_rate))
    test_ = ai >= (thresh - 1)
    prt = (thresh - 1) / n_folds
    parts = [(int(j * prt), int((j + 1) * prt)) for j in range(n_folds)]
    train_ = [(pt[0] <= ai) * (ai < pt[1]) for pt in parts]
    test = numpy.zeros(shape=(N,), dtype=bool)
    train = []
    for j in range(n_folds):
        ap = numpy.zeros(shape=(N,), dtype=bool)
        ap[d0] = train_[j]
        train.append(ap)
    test[d0] = test_
    return train, test


class TimeSerialCV:

    def __init__(self, d0, n_folds=5, test_rate=0.2):
        self.train, self.test = time_serial(d0, n_folds, test_rate)
        self.n_folds = n_folds

    def triple(self, j):
        if j < (self.n_folds - 1):
            return self.train[j], self.train[(j + 1)], self.test
        if j == (self.n_folds - 1):
            return self.train[j], self.train[j], self.test
