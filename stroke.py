#%%[Markdown]

# T2 Project - Stroke Detection
#%%
# Importing packages
import os
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sn 

#%%
# Loading the data set
df = pd.read_csv('stroke dataset.csv')
# %%
df.info()
df.head()

# Starting inital EDA 
# -'id' column is randomly generated int digits, hence not required.    
# -Converting 'age' column to type int64. 
# -'avg_glucose_level' is of type float64.
# -'Stroke' column is the target variable or y. 
# -Except for 'bmi' and 'avg_glucose_level', all other columns are categorical in nature.

# %%
df.drop('id', axis = 1, inplace = True)
# %%

'''
Now converting categorical columns to ordinal/numeric values.
    gender             |Male|Female --> 1|0
    hypertension       |1|0  
    heart_disease      |1|0  
    ever_married       |Yes|No --> 1|0 
    work_type          |Private|Self-emp.|children|Govt_job|Never_worked --> 0|1|2|3|4 
    Residence_type     |Urban|Rural --> 1|0 
    smoking_status     |never_smoked|unknown|formerly_smoked|smokes --> 0|1|2|3
 
'''
#%%
df['gender'].replace(to_replace=['Male', 'Female'], value=['1', '0'], inplace=True)
df['ever_married'].replace(to_replace=['Yes', 'No'], value=['1', '0'], inplace=True)
df['work_type'].replace(to_replace=['Private', 'Self-employed', 'children', 'Govt_job', 'Never_worked'], value=['0', '1', '2', '3', '4'], inplace=True)
df['Residence_type'].replace(to_replace=['Urban', 'Rural'], value=['1', '0'], inplace=True)
df['smoking_status'].replace(to_replace=['never smoked', 'Unknown', 'formerly smoked', 'smokes'], value=['0', '1', '2', '3'], inplace=True)

df.head()

#%%
# BMI column has 201 NaN values. Dropping rows with NaN values. 
# drop all rows that have any NaN values
df = df.dropna()

# reset index of DataFrame
df = df.reset_index(drop=True)

df.shape
#%%
value_count_stroke = df['stroke'].value_counts()
print(value_count_stroke)

# 0    4700
# 1     209
# Name: stroke, dtype: int64

# From initial analysis, the dataset seems to be highly unbalanced. 
# There are 4700 cases without a stroke and 209 cases with a stroke among the participant list. 
#%%
# Further subdiving the dataset into male and female sets. 

grouped = df.groupby(df['gender'])
male_stroke = grouped.get_group('1')
female_stroke = grouped.get_group('0')
# %%
print(male_stroke['stroke'].value_counts())
# 0    1922
# 1     89
# Name: stroke, dtype: int64
# Extremely unbalanced set. 108 men with stroke and 2007 without stroke. 
print(female_stroke['stroke'].value_counts())
# 0    2777
# 1     120
# Name: stroke, dtype: int64
# Extremely unbalanced set. 141 women with stroke and 2853 without stroke. 


# %%
# Above analysis suggests that the data set is highly unbalanced. 
# This will require balancing the dataset (target variable) before any furthur tests or analysis can be conducted! 

# Balancing the data set using SMOTE: Synthetic Minority Oversampling Technique. Using this method, as the name suggests, the minority target variable is oversampled using random values. The technique uses the concept of K-NN or K neareast neighbors to intelligently generate synthetic data which resembles the values or shape of the outnumbered data instead of directly copying or reusing pre-existing values. 
# For more info: https://github.com/scikit-learn-contrib/imbalanced-learn

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# separating the target variable from the main data set. 

X = df.drop('stroke', axis = 'columns') # regressor data set
y = df['stroke'] # target variable data set

print(y.value_counts())
# 0    4700
# 1     209
# Name: stroke, dtype: int64
#%%
# install imbalanced-learn package that has SMOTE. 
# pip install imbalanced-learn --user
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy = 'minority')
X_sm, y_sm = smote.fit_resample(X, y)
# %%
