#

from torch import nn

from mpydge.models.some import Hydrogenium
from mpydge.utils.torchy import ERF


class LogitModel(Hydrogenium):

    def __init__(self, data):
        super().__init__(data=data, layers_dimensions=[2], activators=[nn.Sigmoid], preprocessor=None,
                         embeddingdrop=0.0, activators_args={}, interprocessors=None, interdrops=None, postlayer=None)


class ProbitModel(Hydrogenium):

    def __init__(self, data):
        super().__init__(data=data, layers_dimensions=[2], activators=[ERF], preprocessor=None,
                         embeddingdrop=0.0, activators_args={}, interprocessors=None, interdrops=None, postlayer=None)

