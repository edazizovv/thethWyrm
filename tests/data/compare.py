#


#
from seaborn import load_dataset
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


#


#
data = load_dataset(name='diamonds')

target = 'price'
quantitative = ['carat', 'depth', 'table', 'x', 'y', 'z']

sample_point = 0.8
data_train_ix_mask = [x <= int(data.shape[0] * sample_point) for x in range(data.shape[0])]
data_test_ix_mask = [int(data.shape[0] * sample_point) < x for x in range(data.shape[0])]
X_train = data.loc[data_train_ix_mask, quantitative].values
X_test = data.loc[data_test_ix_mask, quantitative].values
Y_train = data.loc[data_train_ix_mask, target].values
Y_test = data.loc[data_test_ix_mask, target].values

model = XGBRegressor()
model.fit(X=X_train, y=Y_train)
Y_train_hat = model.predict(X_train)
Y_test_hat = model.predict(X_test)

ass_train = r2_score(y_true=Y_train, y_pred=Y_train_hat)
ass_test = r2_score(y_true=Y_test, y_pred=Y_test_hat)
