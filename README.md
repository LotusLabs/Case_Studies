# Case_Studies
This repository contains the analysis of case studies for LotusLabs.

## Medical costs insurance claims
In this project, we will discuss how to predict the insurance claim. 
We take a sample of 1338 data which consists of the following features:

- age : age of the policyholder
- sex: gender of policy holder (female=0, male=1)
- bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 25
- children: number of children/dependents of the policyholder
- smoker: smoking state of policyholder (non-smoke=0;smoker=1)
- region: the residential area of policyholder in the US (northeast=0, northwest=1, southeast=2, southwest=3)
- charges: individual medical costs billed by health insurance
- insuranceclaim – The labeled output from the above features, 1 for valid insurance claim / 0 for invalid.

### Important files
- `analysis.py`, file containing the analysis
- `insurance_claims.csv`, data set file
- in __models__ you will find the parameters of a neural network model

## Retail Sales Forecasting
In this project we propose a neural network architecture to forecast retail sales. The dataset can be found https://www.kaggle.com/manjeetsingh/retaildataset.

The dataset is divided in three tables: Stores, Features and Sales. The stores table contains anonymised information about the 45 stores, indicating the type and size of each store. The features table contains additional data related to the store, department, and regional activity for the given dates. MarkDown data is only available after Nov 2011, and is not available for all stores all the time.

### Features table
- `Store`	the store number
- `Date`	week
- `Temperature`	average temperature in the region
- `FuelPrice`	cost of fuel in the region
- `MarkDown1-5`	anonymized data related to promotional markdowns
- `CPI`	the consumer price index
- `Unemployment`	the unemployment rate
- `IsHoliday`	whether the week is a special holiday week

### Sales table
- `Store`	the store number
- `Dept`	the department number
- `Date`	the week
- `WeeklySales`	sales for the given department in the given store
- `IsHoliday`	whether the week is a special holiday week

### Important files
- In __src__ / __data__ you will find `data-analysis.py` to create the dataset. Remember to download the model from kaggle first.
- In __src__ / __models__ you will find autoregressive models in `ar_models.py`, machine learning models in `ml.py` and a neural network in `nn.py`


## Churn prediction
In the case of a churn prediction model, there is an imbalance between classes Churned and Not Churned. Usually the latter being a majority class. Upsampling and downsampling can help.

Accuracy in general is not a good metric for unbalanced datasets, because the model will predict with high accuracy the the customer is not churning, but that doesn't help us in predicting if a customer will churn.

Two important mentrics considered here are precision and recall.
Precision is: TP / (TP + FP)
Recall is:    TP / (TP + FN)

- High precision means few false positives: not so many non churners falsely classified as churners.
- High recall means few false negatives: not so many churners falsely classified as non churners. So correctly classifies churners.


The choice depends of course on the business objective and costs:
- If keeping existing customers which are potential churners is more expensive than the value of acquiring customers -> your model should have high precision
- If acquiring customers is more expensive than an offer to keep existing customers -> you want your model to have high recall

The area under a ROC curve is also a good measure of model performances. Often, stakeholders are interested in a single metric that can quantify model performance. The AUC is one metric you can use in these cases, and another is the F1 score, which is calculated as below:

2 * (precision * recall) / (precision + recall)

The advantage of the F1 score is it incorporates both precision and recall into a single metric, and a high F1 score is a sign of a well-performing model, even in situations where you might have imbalanced classes.

### Data

Each row represents a customer, each column contains customer’s attributes described on the column Metadata.

The data set includes information about:

- Customers who left within the last month – the column is called Churn
- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
- Demographic info about customers – gender, age range, and if they have partners and dependents

Data Source: https://www.kaggle.com/blastchar/telco-customer-churn

### Important files

- `churn_analysis.py` python file runnable with all the analysis
- `churn_analysis.ipynb` jupyter file containing the analysis
- `tpot_model_churn.py` contains the parameters of an AutoML model run with the TPOT library
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` contains the dataset 