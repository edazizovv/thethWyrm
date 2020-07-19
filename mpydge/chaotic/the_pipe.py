#


#


#
class SimplePipe:

    def __init__(self, data, listed):
        self.data = data
        self.listed = listed

    def go_fit(self):
        data = self.data
        for j in numpy.arange(start=0, step=1, stop=len(self.listed)):
            if self.listed[j].__class == 'model':
                self.listed[j].data = data
                if self.listed[j].__do == 'fit':
                    self.listed[j].fit()
                elif self.listed[h].__do == 'melt_coeff':
                    self.listed[j].melt_coeff()
                data = self.listed[j].predict()     # buggy
            elif self.listed[j].__class == 'transformer':
                self.listed[j].data = data
                self.listed[j].fit()
                data = self.listed[j].transform(data)
        return data

    def go_show(self):
        data = self.data
        for j in numpy.arange(start=0, step=1, stop=len(self.listed)):
            if self.listed[j].__class == 'model':
                self.listed[j].data = data
                data = self.listed[j].predict()     # buggy
            elif self.listed[j].__class == 'transformer':
                self.listed[j].data = data
                data = self.listed[j].transform(data)
        for i in numpy.arange(start=(len(self.listed) - 1), step=-1, stop=-1):
            if self.listed[i].__class == 'model':
                self.listed[i].data = data
                self.listed[i] # ? how to do it ?

    def say_hello_to_daddy(self):
        print('{0}: Oh, hello, okay?'.format(self))

