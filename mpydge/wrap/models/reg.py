#


#
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR as SVR_
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


#
from mpydge.wrap.models.core import SupM1DScikit


#
class OLR(SupM1DScikit):

    def __init__(self, rfe_enabled=False, grid_cv=None, *args, **kwargs):
        super().__init__(_model=LinearRegression, rfe_enabled=rfe_enabled, grid_cv=grid_cv, *args, **kwargs)


class KNR(SupM1DScikit):

    def __init__(self, rfe_enabled=False, grid_cv=None, *args, **kwargs):
        super().__init__(_model=KNeighborsRegressor, rfe_enabled=rfe_enabled, grid_cv=grid_cv, *args, **kwargs)


class DTR(SupM1DScikit):

    def __init__(self, rfe_enabled=False, grid_cv=None, *args, **kwargs):
        super().__init__(_model=DecisionTreeRegressor, rfe_enabled=rfe_enabled, grid_cv=grid_cv, *args, **kwargs)


class ETR(SupM1DScikit):

    def __init__(self, rfe_enabled=False, grid_cv=None, *args, **kwargs):
        super().__init__(_model=ExtraTreesRegressor, rfe_enabled=rfe_enabled, grid_cv=grid_cv, *args, **kwargs)


class RFR(SupM1DScikit):

    def __init__(self, rfe_enabled=False, grid_cv=None, *args, **kwargs):
        super().__init__(_model=RandomForestRegressor, rfe_enabled=rfe_enabled, grid_cv=grid_cv, *args, **kwargs)


class SVR(SupM1DScikit):

    def __init__(self, rfe_enabled=False, grid_cv=None, *args, **kwargs):
        super().__init__(_model=SVR_, rfe_enabled=rfe_enabled, grid_cv=grid_cv, *args, **kwargs)


class GBR(SupM1DScikit):

    def __init__(self, rfe_enabled=False, grid_cv=None, *args, **kwargs):
        super().__init__(_model=GradientBoostingRegressor, rfe_enabled=rfe_enabled, grid_cv=grid_cv, *args, **kwargs)


class LBR(SupM1DScikit):

    def __init__(self, rfe_enabled=False, grid_cv=None, *args, **kwargs):
        super().__init__(_model=LGBMRegressor, rfe_enabled=rfe_enabled, grid_cv=grid_cv, *args, **kwargs)


class XBR(SupM1DScikit):

    def __init__(self, rfe_enabled=False, grid_cv=None, *args, **kwargs):
        super().__init__(_model=XGBRegressor, rfe_enabled=rfe_enabled, grid_cv=grid_cv, *args, **kwargs)
