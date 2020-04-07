# mpydge
--
# Data
###Data format
We use a special class to hold all components of data sets, including:
* train sample
* validation sample
* test sample

Each of them contains of:
* categorical fields
* numerical fields
* output fields

So, for us it is a common technique to initialise data with `TheKeeper` data class

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