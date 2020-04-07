#

import numpy
import pandas
import seaborn
from matplotlib import pyplot

import torch
from torch import nn
# from torch.utils.data import random_split

from sklearn.metrics import confusion_matrix, classification_report


# The Model
# https://stackabuse.com/introduction-to-pytorch-for-classification/
# TODO: Solve all notes from the IDE
class Hydrogenium(nn.Module):
    """
    Model architecture:

        -- Embedding
        -- EmbeddingDrop
        -- Preprocessor

        == LAYER X: -- Linear
                    -- Activator
                    -- Interprocessor (optional)
                    -- Interdrop      (optional)
    """

    def __init__(self, data, layers_dimensions,
                 activators, preprocessor=nn.BatchNorm1d, embeddingdrop=0.0, activators_args={}, interprocessors=None,
                 interdrops=None, postlayer=None):
        super().__init__()

        if data.data.categorical_embeddings is not None:
            self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in data.data.categorical_embeddings])
            if preprocessor is not None:
                self.batch_norm_num = preprocessor(data.data.numerical[1])
            self.embedding_dropout = nn.Dropout(embeddingdrop)
            num_categorical_cols = sum((nf for ni, nf in data.data.categorical_embeddings))
        else:
            self.all_embeddings = None
            num_categorical_cols = 0

        all_layers = []
        num_numerical_cols = data.data.numerical[1]
        input_size = num_categorical_cols + num_numerical_cols

        if not isinstance(activators, list):
            activators = [activators] * len(layers_dimensions)
        if not isinstance(activators_args, list):
            activators_args = [activators_args] * len(layers_dimensions)
        if interprocessors is not None and not isinstance(interprocessors, list):
            interprocessors = [interprocessors] * len(layers_dimensions)
        if interdrops is not None and not isinstance(interdrops, list):
            interdrops = [interdrops] * len(layers_dimensions)

        for j in range(len(layers_dimensions)):
            all_layers.append(nn.Linear(input_size, layers_dimensions[j]))
            all_layers.append(activators[j](**activators_args[j]))
            if interprocessors is not None:
                all_layers.append(interprocessors[j](layers_dimensions[j]))
            if interdrops is not None:
                all_layers.append(nn.Dropout(interdrops[j]))
            input_size = layers_dimensions[j]

        if postlayer is not None:
            all_layers.append(postlayer(layers_dimensions[-1], data.data.n_classes))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical, x_numerical):
        if self.all_embeddings is not None:
            embeddings = []
            for i, e in enumerate(self.all_embeddings):
                embeddings.append(e(x_categorical[:, i]))
            x_embedding = torch.cat(embeddings, 1)
            x_embedding = self.embedding_dropout(x_embedding)
        else:
            x_embedding = None

        if self.all_embeddings is not None and self.all_numerical is not None:
            x = torch.cat([x_embedding, x_numerical], 1)
        if self.all_embeddings is None and self.all_numerical is not None:
            x = torch.cat([x_numerical], 1)
        if self.all_embeddings is not None and self.all_numerical is None:
            x = torch.cat([x_embedding], 1)

        x = self.layers(x)
        return x

    def fit(self, categorical_data_train, numerical_data_train, output_data_train, optimiser, loss_function,
            categorical_data_validation, numerical_data_validation, output_data_validation, epochs=500):

        self.epochs = epochs
        self.aggregated_losses = []
        self.validation_losses = []

        for i in range(epochs):
            i += 1
            for phase in ['train', 'validate']:

                if phase == 'train':
                    y_pred = self(categorical_data_train, numerical_data_train)
                    single_loss = loss_function(y_pred, output_data_train)
                else:
                    y_pred = self(categorical_data_validation, numerical_data_validation)
                    single_loss = loss_function(y_pred, output_data_validation)

                optimiser.zero_grad()

                if phase == 'train':
                    train_lost = single_loss.item()
                    self.aggregated_losses.append(single_loss)
                    single_loss.backward()
                    optimiser.step()
                else:
                    validation_lost = single_loss.item()
                    self.validation_losses.append(single_loss)

            if i % 25 == 1:
                print('epoch: {0:3} train loss: {1:10.8f} validation loss: {2:10.8f}'.format(i, train_lost,
                                                                                             validation_lost))
        print('epoch: {0:3} train loss: {1:10.8f} validation loss: {2:10.8f}'.format(i, train_lost, validation_lost))


    def fit_plot(self):

        pyplot.plot(numpy.array(numpy.arange(self.epochs)), self.aggregated_losses, label='Train')
        pyplot.plot(numpy.array(numpy.arange(self.epochs)), self.validation_losses, label='Validation')
        pyplot.legend(loc="upper left")
        pyplot.show()

    def predict(self, categorical_data, numerical_data):

        output = self(categorical_data, numerical_data)
        result = numpy.argmax(output.detach().numpy(), axis=1)

        return result

    def summary(self, categorical_data, numerical_data, output_data, loss_function=None, show_confusion_matrix=True,
                report=False, score=None):

        y_val = self(categorical_data, numerical_data)
        y_hat = self.predict(categorical_data, numerical_data)
        y = output_data.detach().numpy()

        if loss_function is not None:
            print('{0:25}: {1:10.8f}'.format(str(loss_function)[:-2], loss_function(y_val, output_data)))

        if show_confusion_matrix:
            seaborn.heatmap(confusion_matrix(y, y_hat), annot=True)

        if report:
            print(classification_report(y, y_hat))


class Hydrogenium_old(nn.Module):
    """
    Model architecture:

        -- Embedding
        -- EmbeddingDrop
        -- Preprocessor

        == LAYER X: -- Linear
                    -- Activator
                    -- Interprocessor (optional)
                    -- Interdrop      (optional)
    """

    def __init__(self, dimensionality_embedding, dimensionality_numerical, dimensionality_output, layers_dimensions,
                 activators, preprocessor=nn.BatchNorm1d, embeddingdrop=0.0, activators_args={}, interprocessors=None,
                 interdrops=None, postlayer=None):
        super().__init__()

        if dimensionality_embedding is not None:
            self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in dimensionality_embedding])
            if preprocessor is not None:
                self.batch_norm_num = preprocessor(dimensionality_numerical)
            self.embedding_dropout = nn.Dropout(embeddingdrop)
            num_categorical_cols = sum((nf for ni, nf in dimensionality_embedding))
        else:
            self.all_embeddings = None
            num_categorical_cols = 0
        if dimensionality_numerical is not None:
            self.all_numerical = 1
        else:
            self.all_numerical = None

        all_layers = []
        num_numerical_cols = dimensionality_numerical
        input_size = num_categorical_cols + num_numerical_cols

        if not isinstance(activators, list):
            activators = [activators] * len(layers_dimensions)
        if not isinstance(activators_args, list):
            activators_args = [activators_args] * len(layers_dimensions)
        if interprocessors is not None and not isinstance(interprocessors, list):
            interprocessors = [interprocessors] * len(layers_dimensions)
        if interdrops is not None and not isinstance(interdrops, list):
            interdrops = [interdrops] * len(layers_dimensions)

        for j in range(len(layers_dimensions)):
            all_layers.append(nn.Linear(input_size, layers_dimensions[j]))
            all_layers.append(activators[j](**activators_args[j]))
            if interprocessors is not None:
                all_layers.append(interprocessors[j](layers_dimensions[j]))
            if interdrops is not None:
                all_layers.append(nn.Dropout(interdrops[j]))
            input_size = layers_dimensions[j]

        if postlayer is not None:
            all_layers.append(postlayer(layers_dimensions[-1], dimensionality_output))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical, x_numerical):
        if self.all_embeddings is not None:
            embeddings = []
            for i, e in enumerate(self.all_embeddings):
                embeddings.append(e(x_categorical[:, i]))
            x_embedding = torch.cat(embeddings, 1)
            x_embedding = self.embedding_dropout(x_embedding)
        else:
            x_embedding = None

        if self.all_embeddings is not None and self.all_numerical is not None:
            x = torch.cat([x_embedding, x_numerical], 1)
        if self.all_embeddings is None and self.all_numerical is not None:
            x = torch.cat([x_numerical], 1)
        if self.all_embeddings is not None and self.all_numerical is None:
            x = torch.cat([x_embedding], 1)

        x = self.layers(x)
        return x

    def fit(self, categorical_data_train, numerical_data_train, output_data_train, optimiser, loss_function,
            categorical_data_validation, numerical_data_validation, output_data_validation, epochs=500):

        self.epochs = epochs
        self.aggregated_losses = []
        self.validation_losses = []

        for i in range(epochs):
            i += 1
            for phase in ['train', 'validate']:

                if phase == 'train':
                    y_pred = self(categorical_data_train, numerical_data_train)
                    single_loss = loss_function(y_pred, output_data_train)
                else:
                    y_pred = self(categorical_data_validation, numerical_data_validation)
                    single_loss = loss_function(y_pred, output_data_validation)

                optimiser.zero_grad()

                if phase == 'train':
                    train_lost = single_loss.item()
                    self.aggregated_losses.append(single_loss)
                    single_loss.backward()
                    optimiser.step()
                else:
                    validation_lost = single_loss.item()
                    self.validation_losses.append(single_loss)

            if i % 25 == 1:
                print('epoch: {0:3} train loss: {1:10.8f} validation loss: {2:10.8f}'.format(i, train_lost,
                                                                                             validation_lost))
        print('epoch: {0:3} train loss: {1:10.8f} validation loss: {2:10.8f}'.format(i, train_lost, validation_lost))

    def fit_plot(self):

        pyplot.plot(numpy.array(numpy.arange(self.epochs)), self.aggregated_losses, label='Train')
        pyplot.plot(numpy.array(numpy.arange(self.epochs)), self.validation_losses, label='Validation')
        pyplot.legend(loc="upper left")
        pyplot.show()

    def predict(self, categorical_data, numerical_data):

        output = self(categorical_data, numerical_data)
        result = numpy.argmax(output.detach().numpy(), axis=1)

        return result

    def summary(self, categorical_data, numerical_data, output_data, loss_function=None, show_confusion_matrix=True,
                report=False, score=None):

        y_val = self(categorical_data, numerical_data)
        y_hat = self.predict(categorical_data, numerical_data)
        y = output_data.detach().numpy()

        if loss_function is not None:
            print('{0:25}: {1:10.8f}'.format(str(loss_function)[:-2], loss_function(y_val, output_data)))

        if show_confusion_matrix:
            seaborn.heatmap(confusion_matrix(y, y_hat), annot=True)

        if report:
            print(classification_report(y, y_hat))
