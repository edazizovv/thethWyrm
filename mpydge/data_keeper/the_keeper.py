#

import numpy
import torch


class Samples:
    @staticmethod
    def sample(categorical, numerical, output, categorical_embeddings, n_classes, test_partition, validation_partition, verbose=True):

        # Pytorch Data Preparation

        categorical = torch.tensor(categorical, dtype=torch.int64)
        numerical = torch.tensor(numerical, dtype=torch.float)
        output = torch.tensor(output).flatten()

        if verbose:
            print(categorical.shape)
            print(numerical.shape)
            print(output.shape)

        # Train-Test Split

        samples = [int(test_partition * len(output))]
        samples = [len(output) - samples[0]] + samples
        a = numpy.array(numpy.arange(len(output)))
        a_train = numpy.random.choice(a, size=samples[0])
        a_test = numpy.setdiff1d(a, a_train)

        categorical_data_train, categorical_data_test = categorical[a_train], categorical[a_test]
        numerical_data_train, numerical_data_test = numerical[a_train], numerical[a_test]
        output_data_train, output_data_test = output[a_train], output[a_test]

        if verbose:
            print(categorical_data_train.shape, categorical_data_test.shape)
            print(numerical_data_train.shape, numerical_data_test.shape)
            print(output_data_train.shape, output_data_test.shape)

        # Train-Validation Split

        samples = [int(validation_partition * len(output_data_train))]
        samples = [len(output_data_train) - samples[0]] + samples
        a = numpy.array(numpy.arange(len(output_data_train)))
        a_train = numpy.random.choice(a, size=samples[0])
        a_validation = numpy.setdiff1d(a, a_train)

        categorical_data_train, categorical_data_validation = categorical[a_train], categorical[a_validation]
        numerical_data_train, numerical_data_validation = numerical[a_train], numerical[a_validation]
        output_data_train, output_data_validation = output[a_train], output[a_validation]

        if verbose:
            print(categorical_data_train.shape, categorical_data_validation.shape, categorical_data_test.shape)
            print(numerical_data_train.shape, numerical_data_validation.shape, numerical_data_test.shape)
            print(output_data_train.shape, output_data_validation.shape, output_data_test.shape)

        train = DataFormats(categorical_data_train, numerical_data_train, output_data_train)
        validation = DataFormats(categorical_data_validation, numerical_data_validation, output_data_validation)
        test = DataFormats(categorical_data_test, numerical_data_test, output_data_test)

        return train, validation, test, categorical_embeddings, n_classes


class DataFormats:
    def __init__(self, categorical=None, numerical=None, output=None, categorical_embeddings=None, n_classes=None):
        self.categorical = categorical
        self.numerical = numerical
        self.output = output
        self.categorical_embeddings = categorical_embeddings
        self.n_classes = n_classes

    def gain_all(self):
        return {'categorical': self.categorical, 'numerical': self.numerical, 'output': self.output, 'categorical_embeddings': self.categorical_embeddings, 'n_classes': self.n_classes}


class DataRoles:
    def __init__(self, train=None, validation=None, test=None, categorical_embeddings=None, n_classes=None, non_sampled=None):

        if non_sampled is None:
            self.train = train
            self.validation = validation
            self.test = test
            self.categorical_embeddings = categorical_embeddings
            self.n_classes = n_classes
        else:
            self.train, self.validation, self.test, self.categorical_embeddings, self.n_classes = Samples.sample(**non_sampled.gain_all())


class Medium:
    def __init__(self, data_frame, target, embedding_strategy='default', embedding_explicit=None):
        self.data_frame = data_frame
        if isinstance(target, list):
            self.target = target
        else:
            self.target = [target]
        self._embedding_strategy = embedding_strategy
        self._embedding_explicit = embedding_explicit

    @property
    def data(self):
        data = DataFormats()
        data.categorical = numpy.stack([self.data_frame[col].cat.codes.values for col in self.data_frame.columns.values if (self.data_frame[col].dtype.name == 'category') and (col not in self.target)], axis=1)
        data.numerical = numpy.stack([self.data_frame[col].cat.codes.values for col in self.data_frame.columns.values if (self.data_frame[col].dtype.name == 'float64') and (col not in self.target)], axis=1)
        data.output = self.data_frame[self.target].values
        if self._embedding_strategy == 'default':
            data.categorical_embeddings = [(len(self.data_frame[col].cat.categories), min(50, (len(self.data_frame[col].cat.categories) + 1) // 2)) for col in self.data_frame.columns.values if (self.data_frame[col].dtype.name == 'category') and (col not in self.target)]
        elif self._embedding_strategy is None:
            data.categorical_embeddings = None
        else:
            data.categorical_embeddings = self._embedding_explicit
        if self.data_frame[self.target[0]].dtype.name == 'category':
            data.n_classes = self.data_frame[self.target[0]].cat.categories.values.shape[0]
        else:
            data.n_classes = None
        data = DataRoles(non_sampled=data)
        return data


