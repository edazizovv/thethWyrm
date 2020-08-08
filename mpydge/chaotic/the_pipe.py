#
import numpy


#
from mpydge.wrap.data import DataHandler


#
class SimplePipe:

    def __init__(self, data_frame, train_d0_mask, val_d0_mask, test_d0_mask,
                 qualitative, quantitative,
                 items, items_args, X_names, Y_names, output_names):

        self.data_frame = data_frame

        self.train_d0_mask = train_d0_mask
        self.val_d0_mask = val_d0_mask
        self.test_d0_mask = test_d0_mask

        self.qualitative = qualitative
        self.quantitative = quantitative

        self.items = items
        self.items_args = items_args
        self.X_names = X_names
        self.Y_names = Y_names
        self.output_names = output_names

        self.pipe = []

    def go_fit(self):
        data_frame = self.data_frame.copy()
        for j in numpy.arange(start=0, step=1, stop=len(self.items)):
            columns = [x for x in self.X_names[j] if x in data_frame.columns.values]
            if self.Y_names[j] in data_frame.columns.values:
                columns = columns + [self.Y_names[j]]
            data_frame_ = data_frame[columns]
            quantitative_ = [z for z in columns if z in self.quantitative]
            qualitative_ = [z for z in columns if z in self.qualitative]
            data = DataHandler(data_frame=data_frame_, target=self.Y_names[j],
                               quantitative=quantitative_, qualitative=qualitative_)
            data.mask = [self.train_d0_mask, None]
            self.pipe.append(self.items[j](data, **self.items_args[j]))
            self.pipe[j].fit()
            data_frame.loc[self.train_d0_mask, self.output_names[j]] = self.pipe[j].predict()

    def go_show(self, which):

        if which == 'train':
            target_mask = self.train_d0_mask
        elif which == 'val' or which == 'validation':
            target_mask = self.val_d0_mask
        elif which == 'test':
            target_mask = self.test_d0_mask
        else:
            raise Exception("idk what mask did you mean")

        data_frame = self.data_frame.copy()
        for j in numpy.arange(start=0, step=1, stop=len(self.items)):
            data_frame.loc[target_mask, self.output_names[j]] = self.pipe[j].predict(dim0_mask=target_mask)

        return data_frame.loc[target_mask, :]

    def say_hello_to_daddy(self):
        print('{0}: Oh, hello, okay?'.format(self))

