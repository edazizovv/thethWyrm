#
import numpy


#


#
class SimplePipe:

    def __init__(self, data, items, items_args, X_names, Y_names, output_spec):

        self.data = data

        self.items = items
        self.items_args = items_args

        self.X_names = X_names
        self.Y_names = Y_names
        self.output_spec = output_spec

        self.the_pipe = []

    def output_names(self, j):
        return list(self.output_spec[j].keys())

    def fit(self):

        blade_runner = self.data.copy()

        for j in range(len(self.items)):

            # if self.X_names[j] is None:
            #     self.X_names[j] = [x for x in blade_runner.train.columns.values if x not in self.output_names(j)]
            if isinstance(self.X_names[j], int):
                if self.X_names[j] == 0:
                    self.X_names[j] = [x for x in blade_runner.train.columns.values if x not in self.output_names(j)]
                else:
                    self.X_names[j] = list(self.output_spec[(j + self.X_names[j])].keys())
            if self.Y_names[j] is None:
                # self.Y_names[j] = blade_runner.train.columns.values[0]
                raise Exception("Y_names anyway should be specified as non-None value, set something pls")

            """
            if self.output_spec[j] == '-':
                nexel = list(self.output_spec[(j + 1)].keys())
                self.output_spec[j] = {x: blade_runner.train[x].dtype.name for x in blade_runner.train.columns.values if x not in nexel}
            """

            self.the_pipe.append(self.items[j](**self.items_args[j]))

            print(self.the_pipe[j])

            self.the_pipe[j].fit(blade_runner.train[self.X_names[j]].values, blade_runner.train[self.Y_names[j]].values)

            if self.output_spec[j] is None:
                cols = numpy.array(self.X_names[j])[self.the_pipe[j].support_].tolist()
                self.output_spec[j] = {x: blade_runner.train[x].dtype.name for x in cols}
            blade_runner.train[self.output_names(j)] = self.the_pipe[j].predict(blade_runner.train[self.X_names[j]].values)
            # blade_runner.update_factors(self.output_spec[j])

    def infer(self, on='train'):

        if on == 'train':
            blade_runner = self.data.copy()

            for j in range(len(self.items)):
                blade_runner.train[self.output_names(j)] = self.the_pipe[j].predict(blade_runner.train[self.X_names[j]].values)
                # blade_runner.update_factors(self.output_spec[j])
        elif on == 'validation':
            blade_runner = self.data.copy()

            for j in range(len(self.items)):
                blade_runner.validation[self.output_names(j)] = self.the_pipe[j].predict(blade_runner.validation[self.X_names[j]].values)
                # blade_runner.update_factors(self.output_spec[j])
        elif on == 'test':
            blade_runner = self.data.copy()

            for j in range(len(self.items)):
                blade_runner.test[self.output_names(j)] = self.the_pipe[j].predict(blade_runner.test[self.X_names[j]].values)
                # blade_runner.update_factors(self.output_spec[j])
        else:
            raise Exception("Unacceptable data subset")

        return blade_runner

    def _distorted_infer(self, invertor, diagnozer, juxtapozer):
        _id = self.data.masky_tick()
        _data = self.data.copy()
        while _id == 0:

            blade_runner = self.data.copy()

            for j in range(len(self.items)):
                blade_runner.train[self.output_names[j]] = self.the_pipe[j].predict(blade_runner.train[self.X_names[j]].values)
                # blade_runner.update_factors(self.output_spec[j])

            y_true, y_hat = invertor(self.data, blade_runner)
            diagnozer.fix(y_true, y_hat)
            juxtapozer.fix(self.data, _data)
            # here goes 'diagnose'

            # here goes 'juxtapose'
            _data = self.data.copy()
            _id = self.data.masky_tick()

    def inverse(self):
        raise Exception("Not realized yet")
