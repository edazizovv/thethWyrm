#

import numpy
import pandas
from mpydge.data_keeper.the_keeper import Medium

import torch


def load(verbose=True, test_partition=0.2, validation_partition=0.2):
    # Watch the Data

    categorical_columns = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
    numerical_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

    outputs = ['Exited']

    d = './mpydge/datasets/data/Churn_Modelling.csv'
    data_set = pandas.read_csv(d)

    for category in categorical_columns:
        data_set[category] = data_set[category].astype('category')
    for numeric in numerical_columns:
        data_set[numeric] = data_set[numeric].astype('float64')

    data_set[outputs[0]] = data_set[outputs[0]].astype('category')

    data = Medium(data_frame=data_set, target=outputs)

    return data


def load_old(verbose=True, test_partition=0.2, validation_partition=0.2):
    # Watch the Data

    d = './datasets/data/Churn_Modelling.csv'
    data_set = pandas.read_csv(d)
    data = Medium(data_set)

    # Data Preprocessing

    categorical_columns = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
    numerical_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

    outputs = ['Exited']

    for category in categorical_columns:
        data_set[category] = data_set[category].astype('category')

    categorical_data = numpy.stack([data_set[col].cat.codes.values for col in categorical_columns], axis=1)
    numerical_data = numpy.stack([data_set[col].values for col in numerical_columns], axis=1)
    output_data = data_set[outputs].values

    if verbose:
        print(categorical_data.shape)
        print(numerical_data.shape)
        print(output_data.shape)

    # Pytorch Data Preparation

    categorical_data = torch.tensor(categorical_data, dtype=torch.int64)
    numerical_data = torch.tensor(numerical_data, dtype=torch.float)
    output_data = torch.tensor(output_data).flatten()

    embedding_sizes = [(len(data_set[col].cat.categories), min(50, (len(data_set[col].cat.categories) + 1) // 2)) for col
                       in categorical_columns]

    if verbose:
        print(categorical_data.shape)
        print(numerical_data.shape)
        print(output_data.shape)

        print(embedding_sizes)

    # Train-Test Split

    samples = [int(test_partition * len(output_data))]
    samples = [len(output_data) - samples[0]] + samples
    a = numpy.array(numpy.arange(len(output_data)))
    a_train = numpy.random.choice(a, size=samples[0])
    a_test = numpy.setdiff1d(a, a_train)

    categorical_data_train, categorical_data_test = categorical_data[a_train], categorical_data[a_test]
    numerical_data_train, numerical_data_test = numerical_data[a_train], numerical_data[a_test]
    output_data_train, output_data_test = output_data[a_train], output_data[a_test]

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

    categorical_data_train, categorical_data_validation = categorical_data[a_train], categorical_data[a_validation]
    numerical_data_train, numerical_data_validation = numerical_data[a_train], numerical_data[a_validation]
    output_data_train, output_data_validation = output_data[a_train], output_data[a_validation]

    if verbose:
        print(categorical_data_train.shape, categorical_data_validation.shape, categorical_data_test.shape)
        print(numerical_data_train.shape, numerical_data_validation.shape, numerical_data_test.shape)
        print(output_data_train.shape, output_data_validation.shape, output_data_test.shape)

    data.train.categorical = categorical_data_train
    data.validation.categorical = categorical_data_validation
    data.test.categorical = categorical_data_test
    data.train.numerical = numerical_data_train
    data.validation.numerical = numerical_data_validation
    data.test.numerical = numerical_data_test
    data.train.output = output_data_train
    data.validation.output = output_data_validation
    data.test.output = output_data_test

    data.embedding_sizes = embedding_sizes
    data.numerical_sizes = numerical_data.shape[1]

    return data

