#%% 
from fastai import *
from fastai.tabular import *
import sklearn as sk
import pandas as pd
import numpy as np

# Loading data
df = pd.read_csv('data/processed/data.csv')

#%% [markdown]
# ## A bit of feature engineering
add_datepart(df,'Date',time=True)
df.head()
df['IsReturn'] = ((df['Weekly_Sales'] <= 0))
df.loc[df['Weekly_Sales'] <= 0] = 1
df['Weekly_Sales'] += 1
df['Weekly_Sales'] = np.log(df['Weekly_Sales'])
print(f"There are {(df['Weekly_Sales'] == 0).sum()} zero valued elements in Weekly Sales")
df.drop('Unnamed: 0',axis=1,inplace=True)

#%% [markdown] 
# ## Train - test split
train = df.loc[df['Year'] != 2012]
test = df.loc[df['Year'] == 2012]

x_train = train.drop('Weekly_Sales',axis=1)
y_train = train['Weekly_Sales']

x_test = test.drop('Weekly_Sales',axis=1)
y_test = test['Weekly_Sales']

#%% [markdown] 
# ## Linear model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred_lr = lr.predict(x_test)
y_train_pred = lr.predict(x_train)

# The coefficients
print('Coefficients: \n', lr.coef_)
# The root mean squared error
rmse_lr = sk.metrics.mean_squared_error(y_test, y_pred_lr) ** 0.5
print(f'Root mean squared error for lr is {rmse_lr}')
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % sk.metrics.r2_score(y_test, y_pred_lr))

#%% [markdown]
# ## Extratree regressor
from sklearn.ensemble import ExtraTreesRegressor

etr = ExtraTreesRegressor(n_estimators=100)
etr.fit(x_train, y_train)
y_pred_etr = etr.predict(x_test)

# The root mean squared error
rmse_etr = sk.metrics.mean_squared_error(y_test, y_pred_etr) ** 0.5
print(f'Root mean squared error for etr is {rmse_etr}')
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % sk.metrics.r2_score(y_test, y_pred_etr))

#%% [markdown]
# ## Random forest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

# The root mean squared error
rmse_rf = sk.metrics.mean_squared_error(y_test, y_pred_rf) ** 0.5
print(f'Root mean squared error for rf is {rmse_rf}')
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % sk.metrics.r2_score(y_test, y_pred_rf))

#%% [markdown]
# ## Lgbm
import numpy as np
import lightgbm as lgb

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

x_train.IsHoliday = x_train.IsHoliday.astype(int)
x_test.IsHoliday = x_test.IsHoliday.astype(int)
x_train.Is_month_start = x_train.Is_month_start.astype(int)
x_test.Is_month_start = x_test.Is_month_start.astype(int)
x_train.Is_month_end = x_train.Is_month_end.astype(int)
x_test.Is_month_end = x_test.Is_month_end.astype(int)
x_train.Is_quarter_end = x_train.Is_quarter_end.astype(int)
x_test.Is_quarter_end = x_test.Is_quarter_end.astype(int)
x_train.Is_quarter_start = x_train.Is_quarter_start.astype(int)
x_test.Is_quarter_start = x_test.Is_quarter_start.astype(int)
x_train.Is_year_end = x_train.Is_year_end.astype(int)
x_test.Is_year_end = x_test.Is_year_end.astype(int)
x_train.Is_year_start = x_train.Is_year_start.astype(int)
x_test.Is_year_start = x_test.Is_year_start.astype(int)
x_train.IsReturn = x_train.IsReturn.astype(int)
x_test.IsReturn = x_test.IsReturn.astype(int)

print('Starting training...')
# train
gbm = lgb.LGBMRegressor(num_leaves=31,
                        learning_rate=0.1,
                        n_estimators=100)
gbm.fit(x_train, y_train,
        eval_set=[(x_test, y_test)],
        eval_metric='l2',
        early_stopping_rounds=5)

print('Starting predicting...')
# predict
y_pred_gbm = gbm.predict(x_test, num_iteration=gbm.best_iteration_)

# The root mean squared error
rmse_rf = sk.metrics.mean_squared_error(y_test, y_pred_gbm) ** 0.5
print(f'Root mean squared error for gbm is {rmse_etr}')
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % sk.metrics.r2_score(y_test, y_pred_gbm))

