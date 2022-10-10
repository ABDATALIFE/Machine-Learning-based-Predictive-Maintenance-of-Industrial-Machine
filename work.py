# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:20:37 2021

@author: Nabs
"""

#%% Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import decimal
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from numpy import mean
from numpy import std
from sklearn import tree
from sklearn.metrics import confusion_matrix


#%% Data Preprocessing
#dataset = pd.read_csv('data_m2.csv')
dataset = pd.read_csv(r"C:\Users\Abdul Basit Aftab\Desktop\MachineLearningThesis\Data + Codes\data_m5.CSV")
dataset['avg'] = dataset.iloc[:,4:7].mean(axis=1)


X_train = dataset.iloc[:1000,[4,5,6,10]].values

X_train = dataset.iloc[:,4:7].values
X_train_new = dataset.iloc[:,13]

#%% For Binning
criteria = [dataset['avg'].between(1, 4),
            dataset['avg'].between(6, 6.3),
            dataset['avg'].between(6.3, 6.50),
            dataset['avg'].between(6.50, 7.0),
            dataset['avg'].between(7.0, 7.3),
            dataset['avg'].between(7.3,7.7),
            dataset['avg'].between(7.7,8),
            dataset['avg'].between(8.0, 8.3),
            dataset['avg'].between(8.3,8.9)]
# criteria = [dataset['avg'].between(1, 4),
#             dataset['avg'].between(6, 6.3),
#             dataset['avg'].between(6.3, 6.5),
#             dataset['avg'].between(6.5, 7),
#             dataset['avg'].between(7, 7.3),
#             dataset['avg'].between(7.3,7.7),
#             dataset['avg'].between(8, 8.3),
#             dataset['avg'].between(8.7,8.9)]

def inrange(a, x, b):
    return min(a, b) < x < max(a, b)


from random import randint
values = [0,10, 20, 30,40,50,60,70,80]
#values1 = [randint(1250,1300),randint(1225,1275), randint(1200,1250), randint(1175,1225),randint(1150,1200),randint(1125,1175),randint(1100,1150),randint(1075,1125),randint(1050,1100)]

#Type_new = pd.Series([])
dataset['qualitytool'] = np.select(criteria, values, 0)
#dataset['Vibration'] = np.select(criteria, values1, 0)
#
#for i in range(dataset.shape[0]):
#    if  inrange(1,dataset["avg"][i],4):
#        Type_new[i]= randint(3,6)
#        
#    elif inrange(6,dataset["avg"][i],6.3):
#        Type_new[i]= randint(4,7)
#
#    elif dataset["avg"][i] in range(6.3,6.50):
#        Type_new[i]= randint(4,7)
#
#    elif dataset["avg"][i] in range(6.50,7.0):
#        Type_new[i]= randint(5,8) 
#        
#    elif dataset["avg"][i] in range(7.0,7.3):
#        Type_new[i]= randint(6,9) 
#        
#    elif dataset["avg"][i] in range(7.3,7.7):
#        Type_new[i]= randint(7,10) 
#        
#    elif dataset["avg"][i] in range(7.7,8):
#        Type_new[i]= randint(5,8) 
#        
#    elif dataset["avg"][i] in range(8.0,8.3):
#        Type_new[i]= randint(5,8) 
#        
#    elif dataset["avg"][i] in range(8.3,8.9):
#        Type_new[i]= randint(5,8) 

# inserting new column with values of list made above       
#dataset.insert(14, "VIB", Type_new)
#%% TRaining Testing Dataset
Y_train = dataset.iloc[:,14].values



X_train = dataset.iloc[:,[4,5,6,10]].values
#X_train = X_train.append(dataset['Vibration']).values
#X_train = dataset.iloc[:,12].values
#X_train= X_train.reshape(-3, 3)
X= X_train
#Y_train= Y_train.reshape(-1, 1)
y= Y_train

#%% Test Train Split
X_train_new , X_test_new , Y_train_new , Y_test_new = train_test_split(X,y,test_size= 0.3, random_state= 0)
X= X_train
#Y_train= Y_train.reshape(-1, 1)
y= Y_train

#%% For accuract score
X_train_newest = dataset.iloc[:,[4,5,6,10]].values
Y_train_newest = dataset.iloc[:,14].values


#%% Logistic Regression

from sklearn.linear_model import LogisticRegression
lm = LogisticRegression(random_state=0)
lm.fit(X_train_new,Y_train_new)


y_pred_lm_split = lm.predict(X_train_newest)
thisisaccuracy_lm = accuracy_score(Y_train_newest, y_pred_lm_split)

print(y_pred_lm_split)
print(thisisaccuracy_lm)
lm_conf_mat = confusion_matrix(Y_train_newest,y_pred_lm_split)
print(lm_conf_mat)



#%% Decision Tree Classifier
clf2 = tree.DecisionTreeClassifier()
clf2 = clf2.fit(X_train_new,Y_train_new)
y_pred_clf2_split = clf2.predict(X_train_newest)
thisisaccuracy_clf2 = accuracy_score(Y_train_newest, y_pred_clf2_split)


print(thisisaccuracy_clf2)
clf2_conf_mat = confusion_matrix(Y_train_newest,y_pred_clf2_split)
print(clf2_conf_mat)



#%%Random Forest Regressor
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=5,random_state=0,bootstrap=True)
rfc.fit(X_train_new,Y_train_new)


y_pred_rfc = rfc.predict(X_test_new)
thisisaccuracy_rfc = accuracy_score(Y_test_new, y_pred_rfc)
print(thisisaccuracy_rfc)



rfc_conf_mat = confusion_matrix(Y_test_new,y_pred_rfc)
print(rfc_conf_mat)



#%% Naive Bayes Theorem
from sklearn.naive_bayes import GaussianNB
clf4 = GaussianNB()
clf4.fit(X_train_new,Y_train_new)
a=clf4.predict(X_test_new)
#y_pred_clf4 = clf4.predict(X_train_newest)


thisisaccuracy_clf4 = accuracy_score(Y_test_new, a)
print(thisisaccuracy_clf4)

clf4_conf_mat = confusion_matrix(Y_test_new,a)
print(clf4_conf_mat)


