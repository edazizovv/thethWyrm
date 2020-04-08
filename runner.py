#

import torch
from torch import nn

from mpydge.datasets.malice_churn import load
from mpydge.models.some import Hydrogenium
from mpydge.models.classic import LogitModel
from mpydge.models.classic import ProbitModel


# Load the Data

data = load()

# Initialise the Model

"""
model = Hydrogenium(dimensionality_embedding=data.embedding_sizes, dimensionality_numerical=data.numerical_sizes, dimensionality_output=2,
                    layers_dimensions=[2], activators=[nn.Sigmoid], preprocessor=None, embeddingdrop=0.0, activators_args={},
                    interprocessors=None, interdrops=None, postlayer=None)
"""
#"""
model = LogitModel(data=data)
#"""
"""
model = ProbitModel(dimensionality_embedding=data.embedding_sizes,
                    dimensionality_numerical=data.numerical_sizes)
"""

print(model)

# Define Optimisation

loss_function = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the Model

epochs = 300
model.fit(optimiser, loss_function, epochs)

model.fit_plot()

model.summary(loss_function=loss_function, show_confusion_matrix=False)

