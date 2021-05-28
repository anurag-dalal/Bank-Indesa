## Problem Statement

### Bank loan defaulter prediction

The Bank Indessa has not done well in last 3 quarters. Their NPAs (Non Performing Assets) have reached all time high. It is starting to lose confidence of its investors. As a result, it’s stock has fallen by 20% in the previous quarter alone.

After careful analysis, it was found that the majority of NPA was contributed by loan defaulters. With the messy data collected over all the years, this bank has decided to use machine learning to figure out a way to find these defaulters and devise a plan to reduce them.

This bank uses a pool of investors to sanction their loans. For example: If any customer has applied for a loan of $20000, along with bank, the investors perform a due diligence on the requested loan application. Keep this in mind while understanding data.

In this challenge, you will help this bank by predicting the probability that a member will default.

## Folder Structure and requirements

The dataset files are contained in ML_Artivatic_dataset folder.
This is done on a conda enviornment in python 3.8.

### Instalation Guides

* Jupyter notebook is used, you can install jupyter via:
```bash
$ conda install -c anaconda jupyter
```
* Then you can clone this repository using git clone, and install requirements.txt
```bash
$ git clone https://github.com/anurag-dalal/Bank-Indesa
$ cd Bank-Indesa
$ pip install -r requirements.txt
```
* Then you can open jupyter by:
```bash
$ jupyter notebook
```
* You can explore two notebook file for data visualizatio, and the XGBoost model.

## Steps followed

### Exploratory Data Analysis and data visualization

Exploratory data analysis (EDA) is used by data scientists to analyze and investigate data sets and summarize their main characteristics, often employing data visualization methods. It helps determine how best to manipulate data sources to get the answers you need, making it easier for data scientists to discover patterns, spot anomalies, test a hypothesis, or check assumptions.

EDA is primarily used to see what data can reveal beyond the formal modeling or hypothesis testing task and provides a provides a better understanding of data set variables and the relationships between them. It can also help determine if the statistical techniques you are considering for data analysis are appropriate.

### Data transformation/cleanup

In this step the extra textual parts are striped off
Convert the datatype to numeric type.

Features where this technique is applied: term, emp_length, last_week_pay, sub_grade

### Missing values imputation

Features where median imputaion is applied: term, loan_amnt, funded_amnt, last_week_pay, int_rate, sub_grade, annual_inc, dti, mths_since_last_delinq, mths_since_last_record, open_acc, revol_bal, revol_util, total_acc, total_rec_int, mths_since_last_major_derog, tot_coll_amt, tot_cur_bal, total_rev_hi_lim, emp_length

Features where zero imputaion is applied: acc_now_delinq, total_rec_late_fee, recoveries, collection_recovery_fee, collections_12_mths_ex_med

### Feature Scaling
XGBoost is not sensitive to monotonic transformations of its features for the same reason that decision trees and random forests are not: the model only needs to pick "cut points" on features to split a node. Splits are not sensitive to monotonic transformations: defining a split on one scale has a corresponding split on the transformed scale.

### Feature Engineering

This is the most important step for any data science based application. We have to select the relevant features.

Feature engineering refers to a process of selecting and transforming variables when creating a predictive model using machine learning or statistical modeling (such as deep learning, decision trees, or regression). The process involves a combination of data analysis, applying rules of thumb, and judgement. It is sometimes referred to as pre-processing, although that term can have a more general meaning.

The data used to create a predictive model consists of an outcome variable, which contains data that needs to be predicted, and a series of predictor variables that contain data believed to be predictive of the outcome variable. For example, in a model predicting property prices, the data showing the actual prices is the outcome variable. The data showing things, such as the size of the house, number of bedrooms, and location, are the predictor variables. These are believed to determine the value of the property.

A "feature" in the context of predictive modeling is just another name for a predictor variable. Feature engineering is the general term for creating and manipulating predictors so that a good predictive model can be created.

More about Feature Engineering can be found in:
* [Link 1](https://www.displayr.com/what-is-feature-engineering/)
* [Link 2](https://medium.com/mindorks/what-is-feature-engineering-for-machine-learning-d8ba3158d97a)

The steps followed in this particular dataset are:
* The column is test_member_id is removed as it is a unique ID for the loan and not hold any relevant information
* loan_status is selected as our target variable or outcome variable.
* member_id, emp_length, loan_amnt, funded_amnt, funded_amnt_inv, sub_grade, int_rate, annual_inc, dti, mths_since_last_delinq, mths_since_last_record, open_acc, revol_bal, revol_util, total_acc, total_rec_int, total_rec_late_fee, mths_since_last_major_derog, last_week_pay, tot_cur_bal, total_rev_hi_lim, tot_coll_amt, recoveries, collection_recovery_fee, term, acc_now_delinq, collections_12_mths_ex_med columns are selected based on EDA.
* A new feature called loan_to_income is created which represent how big the loan a person has taken with respect to his earnings, annual income to loan amount ratio.
* A new feature called bad_state is created, it represents that the repayment was not all hunky-dory.
* A new feature called avl_lines is created, it represents total number of available/unused 'credit lines'.
* A new feature called int_paid is created, it represents interest paid so far.
* A new feature called emi_paid_progress_perc is created, it represents EMIs paid (in terms of percent).
* A new feature called total_repayment_progress is created, it represents total repayments received so far.

So there is a total of 33 features are created for training our model.

### Test Train split
The data is splitted into 30% test and 70% train.

### Model used
A XGBoost model is used. Researching from web, it is found that XGBoost have the best accuracy in task like lone default predictiion.
For Hyper Parameter Optimization RandomizedSearchCV is used to optimize the learning_rate, max_depth, min_child_weight, gamma, colsample_bytree and n_estimators parameters.
RandomizedSearchCV was run with 5 iteration and 5 fold cross validation.

The best parameters that are selected by RandomizedSearchCV:
{'n_estimators': 700, 'min_child_weight': 7, 'max_depth': 20, 'learning_rate': 0.15, 'gamma': 10, 'colsample_bytree': 0.5}

The final ROC score is 92.4865812572%
