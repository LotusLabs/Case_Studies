#%%
# Importing modules and data
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('data/processed/data.csv')

#%% [markdown]
# ## Auto correlation analysis
# First is a good idea to check the autocorrelations in order to define 
# the seasonalities for the model. From the plot we observe lags 1,6,and 52

#%%
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf

weekly_sales = df.groupby(by=['Date'], as_index=False)['Weekly_Sales'].sum()
weekly_sales = weekly_sales.set_index('Date')

fig, axes = plt.subplots(1,2, figsize=(20,5))
plot_acf(weekly_sales, lags=100, ax=axes[0])
plot_pacf(weekly_sales, lags=100, ax=axes[1])
plt.show()

#%%
# Creating AR model with single input
def fit_ar_model(ts, orders):
    
    X=np.array([ ts.values[(i-orders)].squeeze() if i >= np.max(orders) else np.array(len(orders) * [np.nan]) for i in range(len(ts))])
    not_nans = ~np.isnan(X[:,:1]).squeeze()
    Y= ts.values
    lin_reg=LinearRegression()
    lin_reg.fit(X[not_nans],Y[not_nans])
    
    print(lin_reg.coef_, lin_reg.intercept_)
    print('Score factor: %.2f' % lin_reg.score(X[not_nans],Y[not_nans]))
    
    return lin_reg.coef_, lin_reg.intercept_
    
def predict_ar_model(ts, orders, coef, intercept):
    return np.array([np.sum(np.dot(coef, ts.values[(i-orders)].squeeze())) + intercept  if i >= np.max(orders) else np.nan for i in range(len(ts))])

#%% [markdown]
# ## Predicting total weekly sales with AR model with single input

# Fitting with 2010-2011 data
orders=np.array([1,6,52])
coef, intercept = fit_ar_model(weekly_sales[weekly_sales.index<'2012'],orders)
# Predicting 2012
pred=pd.DataFrame(index=weekly_sales[weekly_sales.index>='2011'].index,data=predict_ar_model(weekly_sales[weekly_sales.index>='2011'],orders,coef,intercept))
plt.figure(figsize=(20,5))
plt.plot(weekly_sales[weekly_sales.index<'2012'])
plt.plot(weekly_sales[weekly_sales.index>='2012'],'--',c='blue')
plt.plot(pred,c='red')

plt.show()

#%% [markdown]
# ## Analysing total weekly sales AR 
# We can analyse the sales using RMSE
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

rmse = sqrt(mean_squared_error(weekly_sales[weekly_sales.index>='2012'], pred[pred.index>='2012']))
r2 = r2_score(weekly_sales[weekly_sales.index>='2012'], pred[pred.index>='2012'])
print(f'The Root Mean Squared Error for the AR total sales model is {rmse}')
print(f'The R2 score for the AR total sales model is {r2}')

#%% [markdown]
# ## Predicting single store sales with AR model with external inputs
# Predicting single store sales with AR model with external inputs and summing all the sales to forecast the total sales.

dfext=df.groupby(by=['Date'], as_index=False)[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 
                                                  'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].mean()
dfext = dfext.set_index('Date')
# Adding total sales
weekly_sales = df.groupby(by=['Date'], as_index=False)['Weekly_Sales'].sum()
dfext = pd.merge(dfext,weekly_sales, how='left',on=['Date'])
dfext = dfext.set_index('Date')
# shifting sales of lag 1
dfext['shifted_sales'] = dfext['Weekly_Sales'].shift(-1)
dfext.head()

#%% [markdown]
# ## Quick look at the correlation
import seaborn as sns
corr = dfext.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, 
            annot=True, fmt=".3f",
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()

corr['shifted_sales'].sort_values(ascending=False)

#%%
# Creating AR model with multiple external inputs
def fit_ar_model_ext(ts, orders, ext, fitter=LinearRegression()):
    
    X=np.array([ ts.values[(i-orders)].squeeze() if i >= np.max(orders) else np.array(len(orders) * [np.nan]) for i in range(len(ts))])   
    X = np.append(X, ext.values, axis=1)
    mask = ~np.isnan(X[:,:1]).squeeze()
    Y= ts.values
    fitter.fit(X[mask],Y[mask].ravel())

    print(fitter.coef_, fitter.intercept_)
    print('Score factor: %.2f' % fitter.score(X[mask],Y[mask]))
    
    return fitter.coef_, fitter.intercept_
    
def predict_ar_model_ext(ts, orders, ext, coef, intercept):

    X=np.array([ ts.values[(i-orders)].squeeze() if i >= np.max(orders) else np.array(len(orders) * [np.nan]) for i in range(len(ts))]) 
    X = np.append(X, ext.values, axis=1)
    
    return np.array( np.dot(X, coef.T) + intercept)

#%% [markdown]
# ## Predicting total weekly sales with AR model with external inputs
inputs=dfext[['Unemployment','Fuel_Price','CPI','Temperature',
              'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']]

orders=np.array([1,6,29,46,52])
coef, intercept = fit_ar_model_ext(dfext['Weekly_Sales'],orders,inputs)
pred_ext=pd.DataFrame(index=dfext['Weekly_Sales'].index, data=predict_ar_model_ext(dfext['Weekly_Sales'], orders, inputs, coef, intercept))
plt.figure(figsize=(20,5))
plt.plot(dfext['Weekly_Sales'], 'o')
plt.plot(pred)
plt.plot(pred_ext)
plt.show()

#%% [markdown]
# ## Metrics analysis for External inputs AR
rmse = sqrt(mean_squared_error(dfext['Weekly_Sales'][dfext['Weekly_Sales'].index>='2012'], pred_ext[pred_ext.index>='2012']))
r2 = r2_score(dfext['Weekly_Sales'][dfext['Weekly_Sales'].index>='2012'], pred_ext[pred_ext.index>='2012'])
print(f'The Root Mean Squared Error for the AR total sales model is {rmse}')
print(f'The R2 score for the AR total sales model is {r2}')


