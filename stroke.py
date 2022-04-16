#%%[Markdown]

# T2 Project - Stroke Detection
#%%
# Importing packages
import os
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns 
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
value_count_stroke = df['stroke'].value_counts()
print(value_count_stroke)

# 0    4861
# 1     249
# Name: stroke, dtype: int64

# From initial analysis, the dataset seems to be highly unbalanced. 
# There are 4861 cases without a stroke and 249 cases with a stroke among the participant list. 

# Further subdiving the dataset into male and female sets. 

grouped = df.groupby(df['gender'])
male_stroke = grouped.get_group('Male')
female_stroke = grouped.get_group('Female')
# %%
print(male_stroke['stroke'].value_counts())
# 0    2007
# 1     108
# Name: stroke, dtype: int64
# Extremely unbalanced set. 108 men with stroke and 2007 without stroke. 
print(female_stroke['stroke'].value_counts())
# 0    2853
# 1     141
# Name: stroke, dtype: int64
# Extremely unbalanced set. 141 women with stroke and 2853 without stroke. 
# %%

# This will require balancing the dataset before any furthur tests or analysis can be conducted! 

