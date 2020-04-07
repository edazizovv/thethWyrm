

class TheTuple:
    def __init__(self):
        self.categorical = None
        self.numerical = None
        self.output = None


class TheKeeper:
    def __init__(self, data):
        self.data = data
        self.train = TheTuple()
        self.validation = TheTuple()
        self.test = TheTuple()
        self.embedding_sizes = None
        self.numerical_sizes = None

