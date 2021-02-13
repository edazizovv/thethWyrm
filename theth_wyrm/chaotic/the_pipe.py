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

    def _output_names(self, j):

        return list(self.output_spec[j].keys())

    def _determine_names(self, j, blade_runner):

        if isinstance(self.X_names[j], int):
            if self.X_names[j] == 0:
                self.X_names[j] = [x for x in blade_runner.train.columns.values if x not in self._output_names(j)]
            else:
                self.X_names[j] = list(self.output_spec[(j + self.X_names[j])].keys())
        if self.Y_names[j] is None:
            raise Exception("Y_names anyway should be specified as non-None value, set something pls")

    def _fit_predict(self, j, blade_runner):

        self.the_pipe[j].fit(blade_runner.train[self.X_names[j]].values,
                             blade_runner.train[self.Y_names[j]].values)

        if self.output_spec[j] is None:
            cols = numpy.array(self.X_names[j])[self.the_pipe[j].support_].tolist()
            self.output_spec[j] = {x: blade_runner.train[x].dtype.name for x in cols}
        blade_runner.train[self._output_names(j)] = self.the_pipe[j].predict(blade_runner.train[self.X_names[j]].values)

        return blade_runner

    def _predict(self, j, blade_runner, on='train'):

        if on == 'train':
            blade_runner.train[self._output_names(j)] = self.the_pipe[j].predict(blade_runner.train[self.X_names[j]].values)
        else:
            blade_runner.test[self._output_names(j)] = self.the_pipe[j].predict(blade_runner.test[self.X_names[j]].values)

        return blade_runner

    def _inverse(self, j, blade_runner, on='train'):

        if on == 'train':
            blade_runner.train[self._output_names(j)] = self.the_pipe[j].inverse(blade_runner.train[self.X_names[j]].values)
        else:
            blade_runner.test[self._output_names(j)] = self.the_pipe[j].inverse(blade_runner.test[self.X_names[j]].values)

        return blade_runner

    def fit(self):

        blade_runner = self.data.copy()

        for j in range(len(self.items)):

            # determine names

            self._determine_names(j, blade_runner)

            # append item

            self.the_pipe.append(self.items[j](**self.items_args[j]))

            # fit and predict

            blade_runner = self._fit_predict(j, blade_runner)

            # print current state

            if self.verbose:
                print(self.the_pipe[j])

    def infer(self, on='train'):

        blade_runner = self.data.copy()

        for j in range(len(self.items)):
            blade_runner = self._predict(j, blade_runner, on=on)

        return blade_runner

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
