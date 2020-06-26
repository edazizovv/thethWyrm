#

import torch
from torch import nn

from mpydge.holy.datasets.malice_churn import load
from mpydge.holy.models import Helium


# Load the Data

data = load()
print(data.data_frame.train.categorical.size(), data.data_frame.train.numerical.size())
print(data.data_frame.validation.categorical.size(), data.data_frame.validation.numerical.size())
print(data.data_frame.test.categorical.size(), data.data_frame.test.numerical.size())

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

