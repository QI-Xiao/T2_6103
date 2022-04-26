#%%[Markdown]

# T2 Project - Stroke Detection
#%%
# Importing packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
df[df['gender'] == 'Other']
df = df.drop(3116)
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

df['stroke'].value_counts().plot.bar()
plt.title('Class Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

# 0    4699
# 1     209
# Name: stroke, dtype: int64

# From initial analysis, the dataset seems to be highly unbalanced. 
# There are 4699 cases without a stroke and 209 cases with a stroke among the participant list. 
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

male_stroke['stroke'].value_counts().plot.bar()
plt.title('Male Stroke Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

print(female_stroke['stroke'].value_counts())
# 0    2777
# 1     120
# Name: stroke, dtype: int64
# Extremely unbalanced set. 141 women with stroke and 2853 without stroke. 
female_stroke['stroke'].value_counts().plot.bar()
plt.title('Female Stroke Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

# %%
# Above analysis suggests that the data set is highly unbalanced. 
# This will require balancing the dataset (target variable) before any furthur tests or analysis can be conducted! 

# Balancing the data set using SMOTE: Synthetic Minority Oversampling Technique. Using this method, as the name suggests, the minority target variable is oversampled using random values. The technique uses the concept of K-NN or K neareast neighbors to intelligently generate synthetic data which resembles the values or shape of the outnumbered data instead of directly copying or reusing pre-existing values. 
# For more info: https://github.com/scikit-learn-contrib/imbalanced-learn



# separating the target variable from the main data set. 

X = df.drop('stroke', axis = 'columns') # regressor data set
y = df['stroke'] # target variable data set

print(y.value_counts())
# 0    4699
# 1     209
# Name: stroke, dtype: int64


#%%
# Converting column data type to int64 so ordinal values remain as int and not get float values when SMOTE is being performed as the process will generate synthetic values based on KNN algorithm. 
# For eg: We have to make sure that column values stay IN [1,0] and not something like 0.55 when synthetic values are being set up. 

X['gender'] = X['gender'].astype(np.int64)
X['ever_married'] = X['ever_married'].astype(np.int64)
X['Residence_type'] = X['Residence_type'].astype(np.int64)
X['work_type'] = X['work_type'].astype(np.int64)
X['smoking_status'] = X['smoking_status'].astype(np.int64)

X.info()

#%%
# EDA-Statistical Testing (Welch's T-Test for unequal variance and unbalanced sample sizes)

stroke_yes = df[df['stroke']==1]
stroke_no = df[df['stroke']==0]

#%%
import scipy
from scipy import stats

def welch_dof(x,y):
    dof = (x.var()/x.size + y.var()/y.size)**2 / ((x.var()/x.size)**2 / (x.size-1) + (y.var()/y.size)**2 / (y.size-1))
    print(f"Welch-Satterthwaite Degrees of Freedom= {dof:.4f}")

welch_dof(stroke_yes['age'], stroke_no['age'])

def welch_ttest(x, y): 
    ## Welch-Satterthwaite Degrees of Freedom ##
    dof = (x.var()/x.size + y.var()/y.size)**2 / ((x.var()/x.size)**2 / (x.size-1) + (y.var()/y.size)**2 / (y.size-1))
   
    t, p = stats.ttest_ind(x, y, equal_var = False)
    
    print("\n",
          f"Welch's t-test= {t:.4f}", "\n",
          f"p-value = {p:.4f}", "\n",
          f"Welch-Satterthwaite Degrees of Freedom= {dof:.4f}")

welch_ttest(stroke_yes['age'], stroke_no['age'])  

#%%
# EDA - Logit plots for BMI and glucose level variables 
import seaborn as sns

g = sns.lmplot(x="avg_glucose_level", y="stroke", col="gender", hue="gender", data=df, y_jitter=.02, logistic=True)
# Binomial regression/logistic
g.set(xlim=(40, 270), ylim=(-.05, 1.05))

plt.show()

print("\nReady to continue.")

g = sns.lmplot(x="bmi", y="stroke", col="gender", hue="gender", data=df, y_jitter=.02, logistic=True)
# Binomial regression/logistic
g.set(xlim=(0, 80), ylim=(-.05, 1.05))

plt.show()
#%%
# EDA - Pairs plot
sns.set(style="ticks")

sns.pairplot(df, hue="stroke")
plt.show()

print("\nReady to continue.")
#%%
# EDA - Box plots to show relationships between stroke and work-type & stroke and marriage status 
work_ranking = ["0", "1", "2", "3", "4"]

sns.boxplot(x="work_type", y="bmi", color="b", order=work_ranking, data=df)
plt.title('BMI Distribution based on Work Type')
plt.show()

print("\nReady to continue.")

sns.boxplot(x="ever_married", y="bmi", color="b", data=df)
plt.title('BMI Distribution based on Marital Status')
plt.show()

print("\nReady to continue.")

sns.boxplot(x="Residence_type", y="avg_glucose_level", color="b", data=df)
plt.title('Glucose Level Distribution based on Residence Type')
plt.show()

print("\nReady to continue.")

sns.boxplot(x="Residence_type", y="bmi", color="b", data=df)
plt.title('Glucose Level Distribution based on Residence Type')
plt.show()

print("\nReady to continue.")


#%%
# EDA - Stacked bar charts to visualize stroke  hypertension and heart disease 
pivot_heart = pd.pivot_table(data=stroke_yes, values='stroke', index='heart_disease', columns='gender', aggfunc='count')
ax = pivot_heart.plot.bar(stacked=True)
ax.set_title('Count of Stroke Victims with Heart Disease')
print(pivot_heart)
pivot_hyper = pd.pivot_table(data=stroke_yes, values='stroke', index='hypertension', columns='gender', aggfunc='count')
ax = pivot_hyper.plot.bar(stacked=True)
ax.set_title('Count of Stroke Victims with Hypertension')
print(pivot_hyper)
#%%
# install imbalanced-learn package that has SMOTE. 
# pip install imbalanced-learn --user
import imblearn
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy = 'minority')
X_sm, y_sm = smote.fit_resample(X, y)
# %%

print(y_sm.value_counts())

# 1    4699
# 0    4699
# Name: stroke, dtype: int64

# We now have generated equal number of participants who have a stroke and participants who do not have a stroke. 

# The data set is perfectly balanced now!

# %%
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Creating test/train data sets from the balanced set using sklearn train_test_split: 80% train, 20% test 
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size= 0.2, random_state= 15, stratify=y_sm)

print(y_train.value_counts()) # 80% as training set   
print(y_test.value_counts()) # 20% as test set

# %%
from sklearn.linear_model import LogisticRegression

strokemodel = LogisticRegression(max_iter=1000)

strokemodel.fit(X_train, y_train)

#%%
strokemodel.score(X_test, y_test)
# 0.8191489361702128 The model is nearly 82% effective in identifying or predicting the results of a participant getting a stroke or not. 
# %%
# creating predicted target values using the model for X_test:
y_predict = strokemodel.predict(X_test)

# Now we can compare y_predict(predicted) values with actual y_test(real) values using a confusion matrix:

cm_stroke_model = confusion_matrix(y_test, y_predict)
# array([[751, 189],
#        [151, 789]], dtype=int64)
#%%

# Creating a heatmap of the above confusion matrix for better visualization and understanding:

sn.heatmap(cm_stroke_model, annot = True, fmt="d")
plt.show()

# y axis - truth values
# x axis - predicted values

# The heatmap of the confusion matrix shows that when the values in the y_test data set are compared with the predicted values from y_predict.
# 751 participants were predicted to 'not have a stroke' and 751 time it was predicted right by the model. 189 times the participant 'did not have a stroke' but it was predicted they did. 151 participants were predidcted to 'have a stroke' but in reality they did not suffer from one. 789 times the participants were predicted to 'have a stroke' and 789 times they got one. 

print(classification_report(y_test, y_predict))

#              precision    recall  f1-score   support
#
#           0       0.83      0.80      0.82       940
#           1       0.81      0.84      0.82       940
#
#    accuracy                           0.82      1880
#   macro avg       0.82      0.82      0.82      1880
#weighted avg       0.82      0.82      0.82      1880
#%%