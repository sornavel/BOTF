# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 21:27:58 2018

@author: sorna
"""


# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rnd
import seaborn as sns

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from xgboost import XGBClassifier

# importing the dataset
dataset = pd.read_csv('bank.csv')

# drop contact
dataset = dataset.drop('contact',axis=1)

# list data types
dataset.dtypes

# convert few columns to categorical
dataset['job'] = dataset['job'].astype('category')
dataset['marital'] = dataset['marital'].astype('category')
dataset['education'] = dataset['education'].astype('category')
dataset['default'] = dataset['default'].astype('category')
dataset['housing'] = dataset['housing'].astype('category')
dataset['loan'] = dataset['loan'].astype('category')
dataset['month'] = dataset['housing'].astype('category')
dataset['poutcome'] = dataset['poutcome'].astype('category')
dataset['deposit'] = dataset['deposit'].astype('category')

# convert categorical columns to numeric
cat_columns = dataset.select_dtypes(['category']).columns
dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)

# preview first 5 rows
dataset.head()

# convert age to age bands
dataset.loc[ dataset['age'] <= 20, 'age'] = 0
dataset.loc[(dataset['age'] > 20) & (dataset['age'] <= 40), 'age'] = 1
dataset.loc[(dataset['age'] > 40) & (dataset['age'] <= 60), 'age'] = 2
dataset.loc[(dataset['age'] > 60) & (dataset['age'] <= 80), 'age'] = 3
dataset.loc[ dataset['age'] > 80, 'age']

# Create x and y dataframes for processing
x = dataset.drop('deposit',axis=1)
y = dataset['deposit']

# splitting the dataset to training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# preview training dataset
X_train.describe()

# feature scaling
from sklearn.preprocessing import StandardScaler

# save X_test
X_temp = X_test

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# apply decision tree classifier algorithm
classifier= DecisionTreeClassifier("entropy",random_state = 0)
classifier.fit(X_train,y_train)

# predict using test dataset
y_pred=classifier.predict(X_test)

# create confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# calculate the fpr and tpr 
from sklearn import metrics 
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
print (auc(false_positive_rate, true_positive_rate))
print (roc_auc_score(y_test, classifier.predict_proba(X_test)[:,1]))
roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

# plot AUC ROC curve
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# write to output file
output_df = pd.DataFrame(y_pred, X_temp)
output_df.to_csv("c:/users/sorna/output_file_v1.csv")