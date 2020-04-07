# mpydge

# Data
We use a special class to hold all components of data sets, including:
* train sample
* validation sample
* test sample

Each of them contains of:
* categorical fields
* numerical fields
* output fields

It is a convention for us to initialise data with `TheKeeper` data class

`class mpydge.data_keeper.the_keeper.TheKeeper(data=None)`

As you can see, currently it does not have any strong requirements to be initialised. Just simply write

```
from mpydge.data_keeper.the_keeper import TheKeeper
data = TheKeeper()
```

and the instance will be initialised. Then you can consequently set all needed components

For example, train sample:

```
data.train.categorical = my_categorical_data_for_train
data.train.numerical = my_numerical_data_for_train
data.train.output = my_output_data_for_train
```

And so validation and test:

```
data.test.numerical = ...
data.validation.output = ...
```

# Models
All models have unified interface and are applied with several steps. 

1. Initialisation. In this step you set an instance of model class
2. Fit. Now you train the model instance and yield predictive model with training/validation results
3. Inference. Here you use the trained model with any data -- either already used  or new (for example, with test sample)

Notice, that models are data-centric, what means that their concept is **model for dataset**:
in the initialisation step you set key data options (e.g. dimensionality)
and pass training options (e.g. loss function) only in fitting step.
Compare this with, for example, Scikit-Learn interface which is model-centric
(**data for model**): there in initialisation step you set key model options
(e.g. loss function) while passing data only at fitting stage

Lets start with an example LogitModel (suppose we already have out `data` variable):

```buildoutcfg
from mpydge.models.classic import LogitModel

model = LogitModel(dimensionality_embedding=data.embedding_sizes,
                   dimensionality_numerical=data.numerical_sizes)
model.fit(categorical_data_train=data.train.categorical, data.train.numerical, data.train.output, optimiser, loss_function, data.validation.categorical, data.validation.numerical, data.validation.output, epochs)
```