
#%%
from IPython import get_ipython

#%% [markdown]
# # Telco customer churn prediction
# 
# ## Context
# "Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs." [IBM Sample Data Sets]
# 
# ## Content
# Each row represents a customer, each column contains customer’s attributes described on the column Metadata.
# 
# The data set includes information about:
# 
# - Customers who left within the last month – the column is called Churn
# - Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# - Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
# - Demographic info about customers – gender, age range, and if they have partners and dependents
# 
# Data Source: https://www.kaggle.com/blastchar/telco-customer-churn

#%%
import pandas as pd

data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.info()


#%%
data['Churn'].value_counts()


#%%
data.groupby(['Churn']).mean()


#%%
data.groupby('Contract')['Churn'].value_counts()


#%%
data.isna().sum()


#%%
data.isnull().sum()

#%% [markdown]
# There are no NAs or null values

#%%
data.isin([' ']).sum()

#%% [markdown]
# There are 11 values in TotalCharges that are just spaces with no meaning for TotalCharges. We can substitute them with NAs.

#%%
import numpy as np

data = data.replace(' ', np.nan)


#%%
data.isin([' ']).sum()


#%%
data = data.dropna()


#%%
data.isnull().sum()


#%%
data["TotalCharges"] = data["TotalCharges"].astype(float)


#%%
data['tenure']


#%%
# just yes and no
categories_no = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport','StreamingTV', 'StreamingMovies']
for el in categories_no : 
    data[el]  = data[el].replace({'No internet service' : 'No'})
    
data["SeniorCitizen"] = data["SeniorCitizen"].replace({1:"Yes",0:"No"})


#%%
# separating categorical and numerical features
Id_col     = ['customerID']
target_col = ["Churn"]
threshold_numerical = 40
cat_features = data.nunique()[data.nunique() < threshold_numerical].keys().tolist()
cat_features = cat_features[:-1]
bin_features   = data.nunique()[data.nunique() == 2].keys().tolist()
num_features   = [x for x in data.columns if x not in cat_features + ['Churn'] + ['customerID']]
multi_features = [i for i in cat_features if i not in bin_features]

#%% [markdown]
# # Feature scaling and encoding

#%%
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# label encoder for binary features
label_enc = LabelEncoder()
for feat in bin_features :
    data[feat] = label_enc.fit_transform(data[feat])
    
# dummies for cat features
data = pd.get_dummies(data = data,columns = multi_features )

# scaling numerical features
std = StandardScaler()
data_scaled = std.fit_transform(data[num_features])
data_scaled = pd.DataFrame(data_scaled,columns=num_features)

data_copy = data.copy()
data = data.drop(columns = num_features,axis = 1)
data = data.merge(data_scaled,left_index=True,right_index=True,how = "left")

#%% [markdown]
# ## There is no redundant information in the scaled dataframe
# 
# No features need to be removed.

#%%
data.corr()

#%% [markdown]
# # Feature engineering
# 
# Create new features using embeddings from FastAI
# 
# Create new features using automatic methods such as Autofeat

#%%
data.isna().sum()
data = data.dropna()

#%% [markdown]
# # Modeling

#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score,roc_curve,scorer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score,recall_score

#splitting train and test data 
train,test = train_test_split(data,test_size = .25 ,random_state = 256)
    
##seperating dependent and independent variables
cols    = [i for i in data.columns if i not in Id_col + target_col]
train_X = train[cols]
train_Y = train[target_col]
test_X  = test[cols]
test_Y  = test[target_col]

#Function attributes
#dataframe     - processed dataframe
#Algorithm     - Algorithm used 
#training_x    - predictor variables dataframe(training)
#testing_x     - predictor variables dataframe(testing)
#training_y    - target variable(training)
#training_y    - target variable(testing)
#cf - ["coefficients","features"](cooefficients for logistic 
                                 #regression,features for tree based models)

#threshold_plot - if True returns threshold plot for model
    
def churn_prediction(algorithm,training_x,testing_x,
                             training_y,testing_y,cols,cf) :
    
    #model
    algorithm.fit(training_x,training_y)
    predictions   = algorithm.predict(testing_x)
    probabilities = algorithm.predict_proba(testing_x)
    #coeffs
    if   cf == "coefficients" :
        coefficients  = pd.DataFrame(algorithm.coef_.ravel())
    elif cf == "features" :
        coefficients  = pd.DataFrame(algorithm.feature_importances_)
        
    column_df     = pd.DataFrame(cols)

    if cf in ['coefficients','features']:
        coef_sumry    = (pd.merge(coefficients,column_df,left_index= True,
                                right_index= True, how = "left"))
        coef_sumry.columns = ["coefficients","features"]
        coef_sumry    = coef_sumry.sort_values(by = "coefficients",ascending = False)
    
    accuracy = accuracy_score(testing_y,predictions)

    print (algorithm)
    print ("\n Classification report : \n",classification_report(testing_y,predictions))
    print ("Accuracy   Score : ",accuracy)
    #confusion matrix
    conf_matrix = confusion_matrix(testing_y,predictions)
    #roc_auc_score
    model_roc_auc = roc_auc_score(testing_y,predictions) 
    print ("Area under curve : ",model_roc_auc,"\n")

    return predictions,probabilities,accuracy,model_roc_auc

#%% [markdown]
# ## Data Oversampling

#%%
# Data undersampling

#from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE

cols    = [i for i in data.columns if i not in Id_col+target_col]

under_X = data[cols]
under_Y = data[target_col]

#Split train and test data
under_train_X,under_test_X,under_train_Y,under_test_Y = train_test_split(under_X,under_Y,
                                                                         test_size = .1 ,
                                                                         random_state = 256)

#oversampling minority class using smote
#under_sampler = NearMiss(random_state = 256)
under_sampler = SMOTE(random_state=256)
under_sampled_X, under_sampled_Y = under_sampler.fit_sample(under_train_X,under_train_Y)
under_sampled_X = pd.DataFrame(data = under_sampled_X,columns=cols)
under_sampled_Y = pd.DataFrame(data = under_sampled_Y,columns=target_col)
###


#%%


#%% [markdown]
# ## Base line - Logistic regression

#%%
logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

# here it is important to test on the original test set
churn_prediction(logit,under_sampled_X,test_X,under_sampled_Y,test_Y,
                         cols,"coefficients")

#%% [markdown]
# ## KNN

#%%
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
churn_prediction(knn,under_sampled_X,test_X,under_sampled_Y,test_Y,
                         cols,"NA")

#%% [markdown]
# ## Random Forests

#%%
from imblearn.ensemble import BalancedRandomForestClassifier

bal_rf = BalancedRandomForestClassifier(n_estimators=100,sampling_strategy='auto')
churn_prediction(bal_rf,train_X,test_X,train_Y,test_Y,
                         cols,"features")


#%%
from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier(n_estimators=500)
churn_prediction(rf,under_sampled_X,test_X,under_sampled_Y,test_Y,
                         cols,"features")

#%% [markdown]
# ## LGBM

#%%
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                        learning_rate=0.5, max_depth=7, min_child_samples=20,
                        min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
                        n_jobs=-1, num_leaves=500, objective='binary', random_state=None,
                        reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
                        subsample_for_bin=200000, subsample_freq=0)

cols = [i for i in data.columns if i not in Id_col + target_col]
churn_prediction(lgbm,under_sampled_X,test_X,under_sampled_Y,test_Y,
                         cols,"features")

#%% [markdown]
# # XGBoost

#%%
from xgboost import XGBClassifier

xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                    colsample_bytree=1, gamma=0, learning_rate=0.9, max_delta_step=0,
                    max_depth = 7, min_child_weight=1, missing=None, n_estimators=500,
                    n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                    reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                    silent=True, subsample=1)


#%%
churn_prediction(xgb,under_sampled_X,test_X,under_sampled_Y,test_Y,
                         cols,"features")

#%% [markdown]
# # Confusion Matrix

#%%
from matplotlib import pyplot as plt
import itertools
import seaborn as sns

lst    = [lgbm,xgb]

length = len(lst)

mods   = ['Lgbm','XGBoost']

fig = plt.figure(figsize=(20,25))
fig.set_facecolor("#F3F3F3")
for i,j,k in itertools.zip_longest(lst,range(length),mods) :
    plt.subplot(4,3,j+1)
    predictions = i.predict(test_X)
    conf_matrix = confusion_matrix(predictions,test_Y)
    sns.heatmap(conf_matrix,
                annot=True,
                xticklabels=["not churn","churn"],
                yticklabels=["not churn","churn"],
                linewidths = 2)
    plt.title(k,color = "b")
   # plt.subplots_adjust(wspace = .3,hspace = .3)

#%% [markdown]
# # ROC Curves

#%%
lst    = [lgbm,xgb]

length = len(lst)

mods   = ['LGBM',
          'XGBoost']

plt.style.use("dark_background")
fig = plt.figure(figsize=(12,16))
for i,j,k in itertools.zip_longest(lst,range(length),mods) :
    qx = plt.subplot(4,3,j+1)
    probabilities = i.predict_proba(test_X)
    predictions   = i.predict(test_X)
    fpr,tpr,thresholds = roc_curve(test_Y,probabilities[:,1])
    plt.plot(fpr,tpr,linestyle = "dotted",
             color = "royalblue",linewidth = 2,
             label = "AUC = " + str(np.around(roc_auc_score(test_Y,predictions),3)))
    plt.plot([0,1],[0,1],linestyle = "dashed",
             color = "orangered",linewidth = 1.5)
    plt.fill_between(fpr,tpr,alpha = .4)
    plt.fill_between([0,1],[0,1])
    plt.legend(loc = "lower right",
               prop = {"size" : 12})
    #qx.set_facecolor("k")
    plt.grid(True,alpha = .15)
    plt.title(k,color = "b")
    plt.xticks(np.arange(0,1,.3))
    plt.yticks(np.arange(0,1,.3))


#%% [markdown]
# # Precision Recall curves

#%%
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


lst    = [lgbm,xgb]

length = len(lst)

mods   = [ 'LGBM Classifier',
          'XGBoost Classifier']

plt.style.use("dark_background")
fig = plt.figure(figsize=(13,17))
fig.set_facecolor("#F3F3F3")
for i,j,k in itertools.zip_longest(lst,range(length),mods) :
    
    qx = plt.subplot(4,3,j+1)
    probabilities = i.predict_proba(test_X)
    predictions   = i.predict(test_X)
    recall,precision,thresholds = precision_recall_curve(test_Y,probabilities[:,1])
    plt.plot(recall,precision,linewidth = 1.5,
             label = ("avg_pcn : " + 
                      str(np.around(average_precision_score(test_Y,predictions),3))))
    plt.plot([0,1],[0,0],linestyle = "dashed")
    plt.fill_between(recall,precision,alpha = .2)
    plt.legend(loc = "upper right",
               prop = {"size" : 10})
    qx.set_facecolor("black")
    plt.grid(True,alpha = .15)
    plt.title(k,color = "b")
    plt.xlabel("recall",fontsize =17,color='k')
    plt.ylabel("precision",fontsize =17,color='k')
    plt.xlim([0.25,1])
    plt.yticks(np.arange(0,1,.3),color='k')
    plt.xticks(color='k')

#%% [markdown]
# # AutoML

#%%
# from tpot import TPOTClassifier

# tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
# tpot.fit(under_sampled_X,under_sampled_Y)
# print(tpot.score(test_X,test_Y))
# tpot.export('tpot_model_churn.py')


#%%

from supervised.automl import AutoML

automl = AutoML(total_time_limit=180*60,top_models_to_improve=3,
                learner_time_limit=240,algorithms=["Xgboost", "RF", "LightGBM"],
                start_random_models=10, hill_climbing_steps=4)
automl.fit(under_sampled_X,under_sampled_Y)

predictions = automl.predict(test_X)


#%%
accuracy = accuracy_score(test_Y.values,predictions['label'].values)

print ("\n Classification report : \n",classification_report(test_Y,predictions['label'].values))
print ("Accuracy   Score : ",accuracy)
#confusion matrix
conf_matrix = confusion_matrix(test_Y,predictions['label'].values)
#roc_auc_score
model_roc_auc = roc_auc_score(test_Y,predictions['label'].values) 
print ("Area under curve : ",model_roc_auc,"\n")


#%%
# pickle automl model

from sklearn.externals import joblib

# save the model to disk
filename = 'automl_churn_telco1.model'
joblib.dump(model, filename)
 
 
# load the model from disk
loaded_model = joblib.load(filename)

#%% [markdown]
# # Second Dataset
# 
# This bank dataset is available here: https://www.kaggle.com/shrutimechlearn/churn-modelling/

#%% [markdown]
# # Third Dataset
# This dataset is available here: https://www.kaggle.com/mahreen/sato2015

