#%% [markdown]
# # Retail Case Study - Data Analysis
# This notebook contains an exploratory data analysis for the retail case study.
#
# ## Data
# Data can be found at this address: https://www.kaggle.com/manjeetsingh/retaildataset

# ### Content 
# You are provided with historical sales data for 45 stores located in different regions - each store contains a number of departments. The company also runs several promotional markdown events throughout the year. These markdowns precede prominent holidays, the four largest of which are the Super Bowl, Labor Day, Thanksgiving, and Christmas. The weeks including these holidays are weighted five times higher in the evaluation than non-holiday weeks.

# Within the Excel Sheet, there are 3 Tabs – Stores, Features and Sales

# ### Stores
# Anonymized information about the 45 stores, indicating the type and size of store

# ### Features
# Contains additional data related to the store, department, and regional activity for the given dates.

# * Store - the store number
# * Date - the week
# * Temperature - average temperature in the region
# * Fuel_Price - cost of fuel in the region
# * MarkDown1-5 - anonymized data related to promotional markdowns. MarkDown data is only available after Nov 2011, and is not available for all stores all the time. Any missing value is marked with an NA
# * CPI - the consumer price index
# * Unemployment - the unemployment rate
# * IsHoliday - whether the week is a special holiday week

# ### Sales
# Historical sales data, which covers to 2010-02-05 to 2012-11-01. Within this tab you will find the following fields:
# * Store - the store number
# * Dept - the department number
# * Date - the week
# * Weekly_Sales -  sales for the given department in the given store
# * IsHoliday - whether the week is a special holiday week


#%%
import pandas as pd
import matplotlib.pyplot as plt
from fastai import *


#%%
features=pd.read_csv('data/raw/Features data set.csv')
sales=pd.read_csv('data/raw/sales data-set.csv')
stores=pd.read_csv('data/raw/stores data-set.csv')

#%% [markdown]
# ## Transforming dates

#%%
features['Date'] = pd.to_datetime(features['Date'])
sales['Date'] = pd.to_datetime(sales['Date'])

#%% [markdown]
# ## Merging data

#%%
df = pd.merge(sales,features,how='left',on=['Store','Date','IsHoliday'])
df = pd.merge(df, stores,how='left',on=['Store'])

#Filling NAs
df=df.fillna(0)

#%% [markdown]
# ## Transforming types

#%%
df['Temperature'] = (df['Temperature']- 32) * 5./9.
types_encoded, types =df['Type'].factorize()
df['Type'] = types_encoded

#%% [mardown]
# ## Checking data consistency

#%%
info = pd.DataFrame(df.dtypes).T.rename(index={0:'column Type'}) 
info = info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))
info = info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.
                                       rename(index={0: 'null values (%)'}))

#%% [markdown]
# ## Graphical analysis

#%%
# Total weekly sales per day for all the stores
df_weekly_sales = df.groupby(by=['Date'], as_index=False)['Weekly_Sales'].sum()

plt.figure(figsize=(20,5))
plt.plot(df_weekly_sales.Date, df_weekly_sales.Weekly_Sales)
plt.show()

#%%
# Saving processed data
df.to_csv('data/processed/data.csv')