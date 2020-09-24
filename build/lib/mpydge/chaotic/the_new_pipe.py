#


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

    @property
    def output_names(self):
        return [list(x.keys()) for x in self.output_spec]

    def fit(self):

        blade_runner = self.data.copy()

        for j in range(len(self.items)):

            self.the_pipe.append(self.items[j](**self.items_args[j]))
            self.the_pipe[j].fit(blade_runner.train[self.X_names[j]].values, blade_runner.train[self.Y_names[j]].values)

            blade_runner.train[self.output_names[j]] = self.the_pipe[j].predict(blade_runner.train[self.X_names[j]].values)
            blade_runner.update_factors(self.output_spec[j])

    def infer(self, on='train'):

        if on == 'train':
            blade_runner = self.data.copy()

            for j in range(len(self.items)):
                blade_runner.train[self.output_names[j]] = self.the_pipe[j].predict(blade_runner.train[self.X_names[j]].values)
                blade_runner.update_factors(self.output_spec[j])
        elif on == 'validation':
            blade_runner = self.data.copy()

            for j in range(len(self.items)):
                blade_runner.validation[self.output_names[j]] = self.the_pipe[j].predict(blade_runner.validation[self.X_names[j]].values)
                blade_runner.update_factors(self.output_spec[j])
        elif on == 'test':
            blade_runner = self.data.copy()

            for j in range(len(self.items)):
                blade_runner.test[self.output_names[j]] = self.the_pipe[j].predict(blade_runner.test[self.X_names[j]].values)
                blade_runner.update_factors(self.output_spec[j])
        else:
            raise Exception("Unacceptable data subset")

        return blade_runner

    def inverse(self):
        raise Exception("Not realized yet")
