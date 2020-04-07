#

from torch import nn

from mpydge.models.some import Hydrogenium
from mpydge.utils import ERF


class LogitModel(Hydrogenium):

    def __init__(self, dimensionality_embedding, dimensionality_numerical):
        super().__init__(dimensionality_embedding=dimensionality_embedding,
                         dimensionality_numerical=dimensionality_numerical,
                         dimensionality_output=2, layers_dimensions=[2], activators=[nn.Sigmoid], preprocessor=None,
                         embeddingdrop=0.0, activators_args={}, interprocessors=None, interdrops=None, postlayer=None)


class ProbitModel(Hydrogenium):

    def __init__(self, dimensionality_embedding, dimensionality_numerical):
        super().__init__(dimensionality_embedding=dimensionality_embedding,
                         dimensionality_numerical=dimensionality_numerical,
                         dimensionality_output=2, layers_dimensions=[2], activators=[ERF], preprocessor=None,
                         embeddingdrop=0.0, activators_args={}, interprocessors=None, interdrops=None, postlayer=None)

