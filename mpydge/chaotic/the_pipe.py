#
import numpy


#


#
class SimplePipe:

    def __init__(self, data, items, items_args, X_names, Y_names, output_spec, verbose=False):

        self.data = data

        self.items = items
        self.items_args = items_args

        self.X_names = X_names
        self.Y_names = Y_names
        self.output_spec = output_spec

        self.the_pipe = []

        self.verbose = verbose

    def output_names(self, j):
        return list(self.output_spec[j].keys())

    def fit(self):

        blade_runner = self.data.copy()

        for j in range(len(self.items)):

            if isinstance(self.X_names[j], int):
                if self.X_names[j] == 0:
                    self.X_names[j] = [x for x in blade_runner.train.columns.values if x not in self.output_names(j)]
                else:
                    self.X_names[j] = list(self.output_spec[(j + self.X_names[j])].keys())
            if self.Y_names[j] is None:
                raise Exception("Y_names anyway should be specified as non-None value, set something pls")

            self.the_pipe.append(self.items[j](**self.items_args[j]))

            if self.verbose:
                print(self.the_pipe[j])

            self.the_pipe[j].fit(blade_runner.train[self.X_names[j]].values, blade_runner.train[self.Y_names[j]].values)

            if self.output_spec[j] is None:
                cols = numpy.array(self.X_names[j])[self.the_pipe[j].support_].tolist()
                self.output_spec[j] = {x: blade_runner.train[x].dtype.name for x in cols}
            blade_runner.train[self.output_names(j)] = self.the_pipe[j].predict(blade_runner.train[self.X_names[j]].values)

    def infer(self, on='train'):

        if on == 'train':
            blade_runner = self.data.copy()

            for j in range(len(self.items)):
                blade_runner.train[self.output_names(j)] = self.the_pipe[j].predict(blade_runner.train[self.X_names[j]].values)

        elif on == 'test':
            blade_runner = self.data.copy()

            for j in range(len(self.items)):
                blade_runner.test[self.output_names(j)] = self.the_pipe[j].predict(blade_runner.test[self.X_names[j]].values)

        else:
            raise Exception("Unacceptable data subset")

        return blade_runner

    def inverse(self):
        raise Exception("Not realized yet")

    def assess(self, assessor, on, target, where='inner'):
        if where == 'inner':
            if on == 'train':
                bench = self.data.train[target]
                peppa = self.infer(on=on).train[target]
            else:
                bench = self.data.test[target]
                peppa = self.infer(on=on).test[target]
            return assessor(y_true=bench.values, y_hat=peppa.values)
        elif where == 'outer':
            raise Exception("Not realised")
        else:
            raise Exception("Unacceptable plane")
