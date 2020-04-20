#

import torch
from torch import nn

from mpydge.datasets.malice_churn import load
from mpydge.models.some import Hydrogenium
from mpydge.models.classic import LogitModel
from mpydge.models.classic import ProbitModel
from mpydge.models.some import Helium


# Load the Data

data = load()
print(data.data.train.categorical.size(), data.data.train.numerical.size())
print(data.data.validation.categorical.size(), data.data.validation.numerical.size())
print(data.data.test.categorical.size(), data.data.test.numerical.size())

# Initialise the Model

"""
model = Hydrogenium(data=data,
                    layers_dimensions=[2], activators=[nn.Sigmoid], preprocessor=None, embeddingdrop=0.0, activators_args={},
                    interprocessors=None, interdrops=None, postlayer=None)
"""
"""
model = LogitModel(data=data)
"""
"""
model = ProbitModel(data=data)
"""
#"""
model = Helium(data=data,
               layers_dimensions=[2], tau=[0.1], activators=[nn.Identity], preprocessor=None, embeddingdrop=0.0, activators_args={},
               interprocessors=None, interdrops=None, postlayer=None)
#"""

print(model)

# Define Optimisation

loss_function = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the Model

epochs = 300
model.fit(optimiser, loss_function, epochs)

model.fit_plot()

model.summary(loss_function=loss_function, show_confusion_matrix=False)

