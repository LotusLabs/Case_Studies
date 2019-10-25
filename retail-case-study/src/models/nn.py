#%% 
from fastai import *
from fastai.tabular import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Loading data
df = pd.read_csv('data/processed/data.csv')

#%%
# A bit of feature engineering
add_datepart(df,'Date',time=True)
df.head()
df['IsReturn'] = ((df['Weekly_Sales'] <= 0))
df.loc[df['Weekly_Sales'] <= 0] = 1
df['Weekly_Sales'] += 1
df['Weekly_Sales'] = np.log(df['Weekly_Sales'])
print(f"There are {(df['Weekly_Sales'] == 0).sum()} zero valued elements in Weekly Sales")
df.drop('Unnamed: 0',axis=1,inplace=True)

# Defining the dependent variable to predict
dep_var = 'Weekly_Sales'

# Dividing categories between continuous and categorical variables
cont_list, cat_list = cont_cat_split(df=df, max_card=20, dep_var=dep_var)
cont_list, cat_list

# cat_names = train.select_dtypes(include=['object']).columns.tolist()
cat_names = ['IsHoliday',
 'Type',
 'Dept',
 'Year',
 'Month',
 'Week',
 'Day',
 'Dayofweek',
 'Is_month_end',
 'Is_month_start',
 'Is_quarter_end',
 'Is_quarter_start',
 'Is_year_end',
 'Is_year_start',
 'Hour',
 'Minute',
 'Second',
 'IsReturn',
 'Dayofyear',
 'Store']

# cont_names = train.select_dtypes(include=[np.number]).columns.tolist()
cont_names = ['Temperature',
 'Fuel_Price',
 'MarkDown1',
 'MarkDown2',
 'MarkDown3',
 'MarkDown4',
 'MarkDown5',
 'CPI',
 'Unemployment',
 'Size',
 'Elapsed']

print("Categorical columns are : ", cat_names)
print('Continuous numerical columns are :', cont_names)


    #%%
# Splitting training and test data
train = df.loc[df['Year'] != 2012]
test = df.loc[df['Year'] == 2012]

test.isnull().sum()

#%% [markdown]
# ## Processing data
# Data is processed  and transformed with filling missing values, categorization and normalization.

procs = [FillMissing, Categorify, Normalize]

# Test tabularlist
test = TabularList.from_df(test, cat_names=cat_names, cont_names=cont_names, procs=procs)

# Train data bunch
data = (TabularList.from_df(train, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs)
                        .split_by_rand_pct(valid_pct = 0.1, seed = 66)
                        .label_from_df(cols = dep_var, label_cls = FloatList, log = False)
                        .add_test(test)
                        .databunch())

data.show_batch(rows=10)

#%%
# Create deep learning model
learn = tabular_learner(data, layers=[100,80,50], emb_drop=0.2, metrics=rmse,y_range=(1,13))
# select the appropriate learning rate
learn.lr_find()

# we typically find the point where the slope is steepest
learn.recorder.plot(suggestion=True)

#%%
learn.model

#%%
# Fit the model based on selected learning rate
from fastai.callbacks import SaveModelCallback
learn.fit_one_cycle(15, 2e-2, callbacks=[SaveModelCallback(learn, every='improvement', monitor='root_mean_squared_error', name='final')])

#%%
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()

#%%
learn.fit_one_cycle(4,max_lr=1e-2)


#%%
from matplotlib import pyplot as plt

plt.figure()
learn.recorder.plot()
plt.savefig('lr.pdf', bbox_inches='tight')

#%%
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
#%%
learn.fit_one_cycle(5,max_lr=2.75e-4)

#%%
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)

#%%
learn.fit_one_cycle(5,max_lr=2e-6)

#%%
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)

#%%
learn.fit_one_cycle(10,max_lr=1e-6,callbacks=[SaveModelCallback(learn, every='improvement', monitor='root_mean_squared_error', name='best')])
#%%
# get predictions
preds, targets = learn.get_preds(DatasetType.Test)
labels = [p[0].data.item() for p in preds]

preds_np = np.array(preds)
test_np = df.loc[df['Year'] == 2012].Weekly_Sales.values

