#


#
from seaborn import load_dataset
from sklearn.metrics import r2_score


#
from mpydge.chaotic.data import DataHandler
from mpydge.chaotic.the_pipe import SimplePipe
from mpydge.wrap.models.reg import XBR
from mpydge.wrap.transformations.transform import LG, EX
from new_insane import PCA, ZerosReductor
from sell_stone import NoRazor


def join_dicts(a, b):
    un = {}
    for kk in a.keys():
        un[kk] = a[kk]
    for kk in b.keys():
        un[kk] = b[kk]
    return un


def r2(y_true, y_hat):
    return r2_score(y_true=y_true, y_pred=y_hat)


#
data = load_dataset(name='diamonds')

model_kwargs = {'rfe_enabled': False, 'grid_cv': None}

embedding_len = 2
embedder_names = ['EMB_{0}'.format(j) for j in range(embedding_len)]
embedder_kwargs = join_dicts({'rfe_cv': False, 'n_components': embedding_len}, {})

target = 'price'
target_sub = 'price_sub'
# qualitative = ['cut', 'color', 'clarity']
qualitative = []
quantitative = ['carat', 'depth', 'table', 'x', 'y', 'z']
"""
X_names = [qualitative + quantitative, embedder_names,
           -1,
           -1]

Y_names = [target, target, target, target]
output_spec = [{x: 'float64' for x in embedder_names}, None, None, {target: 'float64'}]

items = [PCA, ZerosReductor, NoRazor, XBR]
items_kwargs = [embedder_kwargs,
                {},
                {},
                model_kwargs]
"""

X_names = [# target,
           qualitative + quantitative,
           embedder_names,
           -1,
           -1,
           # target_sub
           ]

Y_names = [# target,
           target,
           target,
           target,
           target,
           # target
           ]

output_spec = [# {target_sub: 'float64'},
               {x: 'float64' for x in embedder_names},
               None,
               None,
               {target: 'float64'},
               # {target: 'float64'}
               ]

items = [# LG,
         PCA,
         ZerosReductor,
         NoRazor,
         XBR,
         # EX
         ]

items_kwargs = [# {},
                embedder_kwargs,
                {},
                {},
                model_kwargs,
                # {}
                ]

"""
X_names = [qualitative + quantitative]

Y_names = [target]
output_spec = [{target: 'float64'}]

items = [XBR]
items_kwargs = [model_kwargs]
"""
sample_points = [0.8]

dta = DataHandler(data_frame=data, qualitative=qualitative, quantitative=quantitative,
                  sample_points=sample_points)
dta.sample()


pipe = SimplePipe(data=dta, items=items, items_args=items_kwargs,
                  X_names=X_names, Y_names=Y_names, output_spec=output_spec)

pipe.fit()
ass_train = pipe.assess(assessor=r2, on='train', target=target)
ass_test = pipe.assess(assessor=r2, on='test', target=target)

Y_train_hat = pipe.infer(on='train')
Y_test_hat = pipe.infer(on='test')
