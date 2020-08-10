#


#


#
class SimplePipe:

    def __init__(self, data, items, items_args, X_names, Y_names, output_names):

        self.data = data

        self.items = items
        self.items_args = items_args

        self.X_names = X_names
        self.Y_names = Y_names
        self.output_names = output_names

        self.the_pipe = []

    def fit(self):

        blade_runner = self.data.train

        for j in range(len(self.items)):

            self.the_pipe.append(self.items[j](**self.items_args[j]))
            self.the_pipe[j].fit(blade_runner[self.X_names[j]].values, blade_runner[self.Y_names[j]].values)

            blade_runner[self.output_names[j]] = self.the_pipe[j].predict(blade_runner[self.X_names[j]].values)
            blade_runner.update_factors(self.output_names[j])

    def infer(self, on='train'):

        if on == 'train':
            blade_runner = self.data.train
        elif on == 'validation':
            blade_runner = self.data.validation
        elif on == 'test':
            blade_runner = self.data.test
        else:
            raise Exception("Unacceptable data subset")

        for j in range(len(self.items)):

            blade_runner[self.output_names[j]] = self.the_pipe[j].predict(blade_runner[self.X_names[j]].values)
            blade_runner.update_factors(self.output_names[j])

        return blade_runner

    def inverse(self):
        raise Exception("Not realized yet")
