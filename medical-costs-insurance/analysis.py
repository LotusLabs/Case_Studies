#%% [markdown] Can you accurately predict insurance claims?

# In this project, we will discuss how to predict the insurance claim. 
# We take a sample of 1338 data which consists of the following features:

# age : age of the policyholder
# sex: gender of policy holder (female=0, male=1)
# bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 25
# children: number of children/dependents of the policyholder
# smoker: smoking state of policyholder (non-smoke=0;smoker=1)
# region: the residential area of policyholder in the US (northeast=0, northwest=1, southeast=2, southwest=3)
# charges: individual medical costs billed by health insurance
# insuranceclaim – The labeled output from the above features, 1 for valid insurance claim / 0 for invalid.


#%% [markdown] 
# # Importing Data

# The cost of treatment depends on many factors: diagnosis, type of clinic, city of residence, age and so on. 
# We have no data on the diagnosis of patients. But we have other information that can help us to make a conclusion about the health of patients and practice regression analysis.

import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as pl
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('insurance_claims.csv')

#%%
data.head()
data.isnull().sum()


#%% [markdown]
# # Encoding categorical labels
from sklearn.preprocessing import LabelEncoder

#sex
le = LabelEncoder()
le.fit(data.sex.drop_duplicates()) 
data.sex = le.transform(data.sex)

# smoker or not
le.fit(data.smoker.drop_duplicates()) 
data.smoker = le.transform(data.smoker)

#region
le.fit(data.region.drop_duplicates()) 
data.region = le.transform(data.region)

#%% [markdown]
# # Correlation
# A strong correlation between being a smoker and charges is observed


#sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), square=True, ax=ax)

grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
f, ax = pl.subplots(figsize=(10, 8))
corr = data.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, cmap="YlGnBu")
    ax.set_title('Feature correlation')

#%% [markdown]
# # Distribution for charges
# Types of Distributions: We have a right skewed distribution in which most patients are being charged between  2000− 12000.
# Using Logarithms: Logarithms helps us have a normal distribution which could help us in a number of different ways such as outlier detection, implementation of statistical concepts based on the central limit theorem and for our predictive modell in the foreseen future.

charge_dist = data["charges"].values
logcharge = np.log(data["charges"])

f= pl.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(charge_dist,color='y',ax=ax)
ax.set_title('Distribution of charges')

ax=f.add_subplot(122)
sns.distplot(logcharge,color='r',ax=ax)
ax.set_title('Log Distribution of charges')


#%% [markdown]
# # BMI and Obesity
# First, let's look at the distribution of costs in patients with BMI greater than 30 and less than 30.

pl.figure(figsize=(12,5))
pl.title("Distribution of charges for patients with BMI greater than 30")
ax = sns.distplot(data[(data.bmi >= 30)]['charges'], color = 'm')

pl.figure(figsize=(12,5))
pl.title("Distribution of charges for patients with BMI less than 30")
ax = sns.distplot(data[(data.bmi < 30)]['charges'], color = 'b')


#%% [markdown]
g = sns.jointplot(x="bmi", y="charges", data = data,kind="kde", color="b")
g.plot_joint(pl.scatter, c="w", s=30, linewidth=1, marker=".")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("BMI", "Charges")


#%% [markdown]
# # Obesity and charges for smokers and non-smokers
# Clear Separation in Charges between Obese Smokers vs Non-Obese Smokers
#In this chart we can visualize how can separate obese smokers and obese non-smokers into different clusters of groups. Therefore, we can say that smoking is a characteristic that definitely affects patient's charges.
pl.figure(figsize=(10,6))
#ax = sns.scatterplot(x='bmi',y='charges',data=data,palette='YlGnBu',hue='smoker')


sns.lmplot(x="bmi", y="charges", hue="smoker", data=data, palette = 'YlGnBu', size = 10)
ax = pl.gca()
ax.set_title('Trends for smokers (1) and non-smokers (0)')
#%% [markdown]
# # Machine learning

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestClassifier


x = data.drop(['insuranceclaim'], axis = 1)
y = data.insuranceclaim

x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 457)

forest = RandomForestClassifier(n_estimators = 200,
                              n_jobs = -1)
forest.fit(x_train,y_train)
forest_train_pred = forest.predict(x_train)
forest_test_pred = forest.predict(x_test)

print('R2 train data: %.3f, R2 test data: %.3f' % (
r2_score(y_train,forest_train_pred),
r2_score(y_test,forest_test_pred)))

from sklearn.metrics import confusion_matrix

confusion_matrix(forest_test_pred, y_test)

#%% [markdown]
# #Classification and ROC analysis
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=6)
classifier = forest

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(x, y):
    probas_ = classifier.fit(x_train, y_train).predict_proba(x_test)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    pl.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
pl.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
pl.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
pl.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

pl.xlim([-0.05, 1.05])
pl.ylim([-0.05, 1.05])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver Operating Characteristic curve')
pl.legend(loc="lower right")
pl.show()

#%% [markdown]
# # Calculating accuracy

from sklearn.metrics import accuracy_score

print(f'The accuracy is {accuracy_score(y_test,forest_test_pred)}')

#%%
importances = classifier.feature_importances_
std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
pl.figure()
pl.title("Feature importances")
pl.bar(range(x.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
pl.xticks(range(x.shape[1]), data.columns)
pl.xlim([-1, x.shape[1]])
pl.show()

for feat, importance in zip(data.columns, classifier.feature_importances_):
    print('feature: {f}, importance: {i}'.format(f=feat, i=importance))


#%% [markdown]
# # Moving to a regression problem
# Now we try to predict the insurance charges, instead of the validity of the
# insurance claim

x = data.drop(['charges'], axis = 1)
y = data.charges

x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 457)

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.compose import TransformedTargetRegressor
forest = RandomForestRegressor(n_estimators = 500,
                              criterion = 'mae',
                              random_state = 1,
                              n_jobs = -1)

def func(x):
    return np.log(x)
def inverse_func(x):
    return np.exp(x)

regr = TransformedTargetRegressor(regressor=forest,
                                  func=func,
                                  inverse_func=inverse_func)
regr.fit(x_train,y_train)
forest_train_pred = regr.predict(x_train)
forest_test_pred = regr.predict(x_test)

y_train = y_train.to_numpy()
print('MAE train data: %.3f, MAE test data: %.3f' % (
metrics.mean_absolute_error(y_train,forest_train_pred),
metrics.mean_absolute_error(y_test,forest_test_pred)))

mae_rf = metrics.mean_absolute_error(y_test,forest_test_pred)

#%%
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(
    regr, x, y, cv=5, scoring='neg_mean_absolute_error')
print(f'Cross-validation scores for RF: {-np.mean(forest_scores)}')

#%%
extra = ExtraTreesRegressor(n_estimators = 300,
                              criterion = 'mae',
                              random_state = 1,
                              n_jobs = -1)
regr = TransformedTargetRegressor(regressor=extra,
                                  func=func,
                                  inverse_func=inverse_func)
regr.fit(x_train,y_train)
extra_train_pred = regr.predict(x_train)
extra_test_pred = regr.predict(x_test)

print('MAE train data: %.3f, MAE test data: %.3f' % (
metrics.mean_absolute_error(y_train,extra_train_pred),
metrics.mean_absolute_error(y_test,extra_test_pred)))

mae_extra = metrics.mean_absolute_error(y_test,extra_test_pred)


#%%
extra_scores = cross_val_score(
    regr, x, y, cv=5, scoring='neg_mean_absolute_error')
print(f'Cross-validation scores for Extra: {-np.mean(extra_scores)}')


#%% [markdown]
# # Deep Neural Network with Embeddings
from fastai.tabular import *
from fastai.data_block import FloatList

data.drop(['insuranceclaim'], axis = 1,inplace=True)

# Dividing categories between continuous and categorical variables
cont_list, cat_list = cont_cat_split(df=data, max_card=10, dep_var='charges')
cont_list, cat_list

#%% [markdown]
# ## Processing data
# Data is processed  and transformed with filling missing values, categorization and normalization.

x = data
y = data.charges

#%%
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
kf.get_n_splits(x)

print(kf)  
mae_nn_scores = []

for train_index, test_index in kf.split(x):

    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    procs = [FillMissing, Categorify, Normalize]

    # Test tabularlist
    test = TabularList.from_df(x_test, cat_names=cat_list, cont_names=cont_list, procs=procs)

    # Train data bunch
    data = (TabularList.from_df(x_train, path='.', cat_names=cat_list, cont_names=cont_list, procs=procs)
                        .split_by_rand_pct(valid_pct = 0.1, seed = 77)
                        .label_from_df(cols = 'charges', label_cls = FloatList, log = True)
                        .add_test(TabularList.from_df(x_test, cat_names=cat_list, cont_names=cont_list, procs=procs))
                        .databunch())

    # Create deep learning model
    learn = tabular_learner(data, layers=[1000,700,500], metrics=mae)
    # select the appropriate learning rate
    # learn.lr_find()
    # we typically find the point where the slope is steepest
    # learn.recorder.plot(suggestion=True)
    learn.fit_one_cycle(20,max_lr=1e-2)
    learn.unfreeze()
    learn.fit_one_cycle(20,max_lr=2e-4)
    learn.unfreeze()
    learn.fit_one_cycle(20,max_lr=8e-6)
    preds = learn.get_preds(ds_type = DatasetType.Test)
    #labels = [p[0].data.item() for p in preds]

    # calculate manually mae with preds and test

    mae_nn = metrics.mean_absolute_error(y_test,np.exp(preds[0]))
    mae_nn_scores.append(mae_nn)

    print(f'Deep NN MAE is:{mae_nn}')

print(f'Final Deep NN MAE is:{np.mean(mae_nn_scores)}')