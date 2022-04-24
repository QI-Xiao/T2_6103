# %%[Markdown]

# T2 Project - Stroke Detection
# %%
# Importing packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

# %%
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
df.drop('id', axis=1, inplace=True)
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
# %%
df['gender'].replace(to_replace=['Male', 'Female'],
                     value=['1', '0'], inplace=True)
df['ever_married'].replace(to_replace=['Yes', 'No'],
                           value=['1', '0'], inplace=True)
df['work_type'].replace(to_replace=['Private', 'Self-employed', 'children',
                        'Govt_job', 'Never_worked'], value=['0', '1', '2', '3', '4'], inplace=True)
df['Residence_type'].replace(to_replace=['Urban', 'Rural'], value=[
                             '1', '0'], inplace=True)
df['smoking_status'].replace(to_replace=['never smoked', 'Unknown',
                             'formerly smoked', 'smokes'], value=['0', '1', '2', '3'], inplace=True)

df.head()

# %%
# BMI column has 201 NaN values. Dropping rows with NaN values.
# drop all rows that have any NaN values
df = df.dropna()

# reset index of DataFrame
df = df.reset_index(drop=True)

df.shape
# %%
value_count_stroke = df['stroke'].value_counts()
print(value_count_stroke)

# 0    4699
# 1     209
# Name: stroke, dtype: int64

# From initial analysis, the dataset seems to be highly unbalanced.
# There are 4699 cases without a stroke and 209 cases with a stroke among the participant list.
# %%
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


# separating the target variable from the main data set.

X = df.drop('stroke', axis='columns')  # regressor data set
y = df['stroke']  # target variable data set

print(y.value_counts())
# 0    4699
# 1     209
# Name: stroke, dtype: int64

# %%
# Converting column data type to int64 so ordinal values remain as int and not get float values when SMOTE is being performed as the process will generate synthetic values based on KNN algorithm.
# For eg: We have to make sure that column values stay IN [1,0] and not something like 0.55 when synthetic values are being set up.

X['gender'] = X['gender'].astype(np.int64)
X['ever_married'] = X['ever_married'].astype(np.int64)
X['Residence_type'] = X['Residence_type'].astype(np.int64)
X['work_type'] = X['work_type'].astype(np.int64)
X['smoking_status'] = X['smoking_status'].astype(np.int64)

X.info()


# %%
# install imbalanced-learn package that has SMOTE.
# pip install imbalanced-learn --user

smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X, y)
# %%

print(y_sm.value_counts())

# 1    4699
# 0    4699
# Name: stroke, dtype: int64

# We now have generated equal number of participants who have a stroke and participants who do not have a stroke.

# The data set is perfectly balanced now!

# %%

# Creating test/train data sets from the balanced set using sklearn train_test_split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_sm, y_sm, test_size=0.2, random_state=15, stratify=y_sm)

print(y_train.value_counts())  # 80% as training set
print(y_test.value_counts())  # 20% as test set

# %%

strokemodel = LogisticRegression(max_iter=1000)

strokemodel.fit(X_train, y_train)

# %%
strokemodel.score(X_test, y_test)
# 0.8191489361702128 The model is nearly 82% effective in identifying or predicting the results of a participant getting a stroke or not.
# %%
# creating predicted target values using the model for X_test:
y_predict = strokemodel.predict(X_test)

# Now we can compare y_predict(predicted) values with actual y_test(real) values using a confusion matrix:

cm_stroke_model = confusion_matrix(y_test, y_predict)
# array([[751, 189],
#        [151, 789]], dtype=int64)
# %%

# Creating a heatmap of the above confusion matrix for better visualization and understanding:

sn.heatmap(cm_stroke_model, annot=True, fmt="d")
plt.show()

# y axis - truth values
# x axis - predicted values

# The heatmap of the confusion matrix compares the values in the y_test data set with the predicted values from y_predict.

# 751 participants were predicted to not have a stroke and 751 participants were predicted correctly.
# 151 participants had a stroke but were predicted incorrectly by the model to not have a stroke. 
# 189 participants did not have a stroke but were predicted incorrectly by the model to have a stroke. 
# 789 participants had a stroke and were predicted correctly by the model to have a stroke.

print(classification_report(y_test, y_predict))

#              precision    recall  f1-score   support
#
#           0       0.83      0.80      0.82       940
#           1       0.81      0.84      0.82       940
#
#    accuracy                           0.82      1880
#   macro avg       0.82      0.82      0.82      1880
# weighted avg       0.82      0.82      0.82      1880

# Accuracy and f1-score of the model is 0.82 or 82% which is pretty decent given that the target variable has been modified.

# %%
# Classification tree model on the same data to compare with the Logit model. 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# %%
tree_model = DecisionTreeClassifier(max_depth=3)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
tree_cm = confusion_matrix(y_test, y_pred_tree)
print(tree_cm)
sn.heatmap(tree_cm, annot = True, fmt = 'd')
plt.show()
print(classification_report(y_test, y_pred_tree))
#
# As max_depth is increased in the tree, the accuracy level of the model gets slightly better each time. 
# This suggests that increasing the depth tends to overfit the model, which makes sense as increasing the tree depth to a value high enough will essentially lead to each data point in the stroke set being a leaf node by itself. 
#   0       1
# 0 [[615 325]
# 1 [ 63 877]]

# 615 participants were predicted to not have a stroke and 615 participants were predicted correctly.
# 63 participants had a stroke but were predicted incorrectly by the model to not have a stroke. 
# 325 participants did not have a stroke but were predicted incorrectly by the model to have a stroke. 
# 877 participants had a stroke and were predicted correctly by the model to have a stroke. 

#               precision    recall  f1-score   support
# 
#            0       0.91      0.65      0.76       940
#            1       0.73      0.93      0.82       940
# 
#     accuracy                           0.79      1880
#    macro avg       0.82      0.79      0.79      1880
# weighted avg       0.82      0.79      0.79      1880

# Overall, accuracy is 79%, which is slightly less than the Logit model. 
#%%
# Running cross-validation on both the models. 
logit_cv = cross_val_score(strokemodel, X_train, y_train, cv= 10, scoring='accuracy')
print(logit_cv)

dtc_cv = cross_val_score(tree_model, X_train, y_train, cv= 10, scoring='accuracy')
print(dtc_cv)
#%%
# Generating the ROC and AUC plots for both the models:
from sklearn.metrics import roc_auc_score, roc_curve

# ROC/AUC for Logit model: 
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_predict)
plt.figure(figsize=(10, 8), dpi=100)
plt.axis('scaled')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("AUC & ROC Curve")
plt.plot(false_positive_rate, true_positive_rate, 'r')
plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightsalmon', alpha=0.6)
plt.text(0.95, 0.05, 'AUC = %0.4f' % roc_auc_score(y_test, y_predict), ha='right', fontsize=12, weight='bold', color='blue')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# AUC = 0.8207

# ROC/AUC for Classification Tree model:
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred_tree)
plt.figure(figsize=(10, 8), dpi=100)
plt.axis('scaled')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("AUC & ROC Curve")
plt.plot(false_positive_rate, true_positive_rate, 'b')
plt.fill_between(false_positive_rate, true_positive_rate, facecolor='steelblue', alpha=0.6)
plt.text(0.95, 0.05, 'AUC = %0.4f' % roc_auc_score(y_test, y_pred_tree), ha='right', fontsize=12, weight='bold', color='blue')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# AUC = 0.7926
# Both models are either over or very close to the 0.80 AUC mark. 
# %%
