
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
dataset = pd.read_csv(r"C:\Users\Abdul Basit Aftab\Desktop\MachineLearningThesis\Data + Codes\data_m5.csv")
##dataset2 = pd.read_csv("C:\\Users\\Nabs\Desktop\\Data Filteration\\data_m2.csv")
dataset['avg'] = dataset.iloc[:,4:7].mean(axis=1)


X_train = dataset.iloc[:,4:7].values

X_train = dataset.iloc[:,4:7].values
X_train_new = dataset.iloc[:,7]








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
# 1 - 4	0
# 5.5 – 5.9	10
# 5.9 – 6.3	20
# 6.3 – 6.7	30
# 6.7 – 7.1	40
# 7.1 – 7.5	50
# 7.5 - 7.9 	60
# 7.9 – 8.3	70
# 8.3 – 8.7	80
# 8.7 – 9.1	90
# 9.1 – 9.5	100


def inrange(a, x, b):
    return min(a, b) < x < max(a, b)


from random import randint
values = [0,10, 20, 30,40,50,60,70,80]
#values1 = [randint(1250,1300),randint(1225,1275), randint(1200,1250), randint(1175,1225),randint(1150,1200),randint(1125,1175),randint(1100,1150),randint(1075,1125),randint(1050,1100)]

#Type_new = pd.Series([])
dataset['qualitytool'] = np.select(criteria, values, 0)
# dataset['Vibration'] = np.select(criteria, values1, 0)

# for i in range(dataset.shape[0]):
#     if  inrange(1,dataset["avg"][i],4):
#         Type_new[i]= randint(3,6)
        
#     elif inrange(6,dataset["avg"][i],6.3):
#         Type_new[i]= randint(4,7)

#     elif dataset["avg"][i] in range(6.3,6.50):
#         Type_new[i]= randint(4,7)

#     elif dataset["avg"][i] in range(6.50,7.0):
#         Type_new[i]= randint(5,8) 
        
#     elif dataset["avg"][i] in range(7.0,7.3):
#         Type_new[i]= randint(6,9) 
        
#     elif dataset["avg"][i] in range(7.3,7.7):
#         Type_new[i]= randint(7,10) 
        
#     elif dataset["avg"][i] in range(7.7,8):
#         Type_new[i]= randint(5,8) 
        
#     elif dataset["avg"][i] in range(8.0,8.3):
#         Type_new[i]= randint(5,8) 
        
#     elif dataset["avg"][i] in range(8.3,8.9):
#         Type_new[i]= randint(5,8) 

# inserting new column with values of list made above       
# dataset.insert(14, "VIB", Type_new)
#%% TRaining Testing Dataset
Y_train = dataset.iloc[:,7].values
#Y_train = 100 - Y_train


X_train = dataset.iloc[:,4:7].values
#X_train = X_train.append(dataset['Vibration']).values
#X_train = dataset.iloc[:,12].values
#X_train= X_train.reshape(-3, 3)
X= X_train
#Y_train= Y_train.reshape(-1, 1)
y= Y_train
aas=X.mean(axis = 1)
#plt.scatter(X[:,1],y) # ploting graph of product quality
plt.scatter(dataset['avg'],dataset['qualitytool']) # plotting of die degradation %
#plt.scatter(X_test_new[:,1],Y_test_new)
plt.title('Current Verses Die Degeradation')
plt.xlabel('Current Ratings , A')
plt.ylabel('Die Degradation , %')
#plt.legend(['Training Dataset','Testing Dataset'],loc=2)
plt.show()
#%% Test Train Split
X_train_new , X_test_new , Y_train_new , Y_test_new = train_test_split(X,y,test_size= 0.3, random_state= 0)
#X_train= X_train.reshape(-1, 1)

X= X_train

#Y_train= Y_train.reshape(-1, 1)
y= Y_train

#%% Data Preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

Y_train = Y_train.reshape(-1,1)
scaledX = scaler.fit_transform(X_train)
scaledY = scaler.fit_transform(Y_train)

#%% For accuract score
X_test_newest = dataset2.iloc[:,4:7].values
#Y_train_newest = dataset.iloc[:83,7].values
#X_test_new= X_test_new.reshape(-1, 1)
#Y_test_new=Y_test_new.reshape(-1,1)
#X_train_newest= X_train_newest.reshape(-1, 1)
#Y_train_newest= Y_train_newest.reshape(-1, 1)
#X_train_new= X_train_new.reshape(-1, 1)
#%% Logistic Regression

from sklearn.linear_model import LogisticRegression
lm = LogisticRegression(random_state=0)
lm.fit(X_train_new,Y_train_new)


y_pred_lm_split = lm.predict(X_test_new)
thisisaccuracy_lm = accuracy_score(Y_test_new, y_pred_lm_split)

print(y_pred_lm_split)
print(thisisaccuracy_lm)
lm_conf_mat = confusion_matrix(Y_test_new,y_pred_lm_split)
print(lm_conf_mat)



#%% Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
clf2 = tree.DecisionTreeClassifier(criterion='entropy',max_depth=20)
#clf2 = tree.DecisionTreeClassifier(max_depth=20)
clf2 = clf2.fit(X_train_new,Y_train_new)
y_pred_clf2_split = clf2.predict(X_test_new)
thisisaccuracy_clf2 = accuracy_score(Y_test_new, y_pred_clf2_split)


print(thisisaccuracy_clf2)
clf2_conf_mat = confusion_matrix(Y_test_new,y_pred_clf2_split)
print(clf2_conf_mat)

#%% Decision tree Trial
import numpy as np
a =np.array([])
b=[]
iter_x_values = np.arange(1,1000)


for i in iter_x_values:
    clf2 = tree.DecisionTreeClassifier(max_depth=i,criterion='gini')
    clf2 = clf2.fit(X_train_new,Y_train_new)
    y_pred_clf2_split = clf2.predict(X_test_new)
    thisisaccuracy_clf2 = accuracy_score(Y_test_new, y_pred_clf2_split)
    b.append(thisisaccuracy_clf2)
    print(i)
    print(thisisaccuracy_clf2)

#%%Random Forest Classifier %91.503
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(criterion='gini',min_samples_leaf=1,min_samples_split=2,n_estimators=46,random_state=0,bootstrap=True)
#random forest is criterion='entropy', Min_samples_leaf=1, min_samples_split=2, n_estimators=10, n_job=2, oob_score=False, random_state=1, warm_start=False.
rfc.fit(X_train_new,Y_train_new)
aaa=rfc.predict_proba(X_train_new)

y_pred_rfc = rfc.predict(X_test_new)
thisisaccuracy_rfc = accuracy_score(Y_test_new, y_pred_rfc)
print(thisisaccuracy_rfc)



rfc_conf_mat = confusion_matrix(Y_test_new,y_pred_rfc)
print(rfc_conf_mat)

    #%% Plotting'
from scipy.interpolate import make_interp_spline

New_Spline = make_interp_spline(iter_x_values,b)

xnew = np.linspace(iter_x_values.min(), iter_x_values.max(), 10)
ynew = New_Spline(xnew)

plt.plot(xnew,ynew)
plt.title('Decison Tree Classification')
plt.xlabel('Number of Nodes, n')
plt.ylabel('Accuracy, %')
plt.show()


#%% Naive Bayes Theorem
from sklearn.naive_bayes import GaussianNB
clf4 = GaussianNB()
#from sklearn.naive_bayes import BernoulliNB
#clf4 = BernoulliNB()
clf4.fit(X_train_new,Y_train_new)
a=clf4.predict(X_test_new)
y_pred_clf4 = clf4.predict(X_train_new)


thisisaccuracy_clf4 = accuracy_score(Y_test_new, a)
print(thisisaccuracy_clf4)

clf4_conf_mat = confusion_matrix(Y_test_new,a)
print(clf4_conf_mat)




#%%PLOTTING DECISION TREE
plt.scatter(X_test_new[:,0],Y_test_new)
plt.scatter(X_test_new[:,0],a)
plt.title('Naive Bayes Algorithm')
plt.xlabel('Current Ratings , A')
plt.ylabel('Die Degradation , %')
plt.legend(['Training Dataset','Testing Dataset'],loc=2)
plt.show()

#%%PLOTTING DECISION TREE
plt.scatter(X_test_new[:,0],a)
plt.scatter(X_test_new[:,0],Y_test_new)
plt.title('Naive Bayes Algorithm')
plt.xlabel('Current Ratings , A')
plt.ylabel('Product quality , %')
plt.legend(['Training Dataset','Testing Dataset'],loc=2)
plt.show()
#%% PLOTTING LOGISTIC REGRESSIOn
plt.scatter(X_test_new[:,1],Y_test_new)
plt.scatter(X_test_new[:,1],y_pred_lm_split)
plt.title('Logistic Regression')
plt.xlabel('Current Ratings , A')
plt.ylabel('Die Degradation , %')
plt.legend(['Training Dataset','Testing Dataset'],loc=2)
plt.show()

#%%PLOTTING RANDOM FOREST
plt.scatter(X_test_new[:,1],Y_test_new)
plt.scatter(X_test_new[:,1],y_pred_rfc)
plt.title('Random Forest Classification')
plt.xlabel('Current Ratings , A')
plt.ylabel('Die Degradation , %')
plt.legend(['Training Dataset','Testing Dataset'],loc=2)
plt.show()
#%%PLOTTING DECISION TREE
plt.scatter(X_test_new[:,1],Y_test_new)
plt.scatter(X_test_new[:,1],y_pred_clf2_split)
plt.title('Decision Tree')
plt.xlabel('Current Ratings , A')
plt.ylabel('Die Degradation , %')
plt.legend(['Training Dataset','Testing Dataset'],loc=2)
plt.show()

#%% Random Forest Changes  #90.84% EFFICIENCY
rfc2 = RandomForestClassifier(n_estimators=30,random_state=1,bootstrap=True)
rfc2.fit(X_train_new,Y_train_new)


y_pred_rfc2 = rfc2.predict(X_test_new)
thisisaccuracy_rfc2 = accuracy_score(Y_test_new, y_pred_rfc2)
print(thisisaccuracy_rfc2)



rfc_conf_mat2 = confusion_matrix(Y_test_new,y_pred_rfc2)
print(rfc_conf_mat2)

#%%PLOTTING RANDOM FOREST
plt.scatter(X_test_new[:,1],Y_test_new)
plt.scatter(X_test_new[:,1],y_pred_rfc2)
plt.title('Random Forest Classification')
plt.xlabel('Current Ratings , A')
plt.ylabel('Die Degradation , %')
plt.legend(['Training Dataset','Testing Dataset'],loc=2)
plt.show()
#%% tree diagram.
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree

fig = plt.figure(figsize=(125,100))
_ = tree.plot_tree(clf2, 
                   feature_names=X_train_new,  
                   class_names=['10','20','30','40','50','60','70','80','90','100'],
                   filled=True)



#%%
import numpy as np
a =np.array([])
b=[]
iter_x_values = np.arange(1,100)


for i in iter_x_values:
    rfc2 = RandomForestClassifier(n_estimators=i,random_state=0,bootstrap=True)
    rfc2.fit(X_train_new,Y_train_new)
    y_pred_rfc2 = rfc2.predict(X_test_new)
    thisisaccuracy_rfc2 = accuracy_score(Y_test_new, y_pred_rfc2)
    b.append(thisisaccuracy_rfc2)
    print(i)
    print(thisisaccuracy_rfc2 )
    
    
    
    


#%% Plotting'
from scipy.interpolate import make_interp_spline

New_Spline = make_interp_spline(iter_x_values,b)

xnew = np.linspace(iter_x_values.min(), iter_x_values.max(), 10)
ynew = New_Spline(xnew)

plt.plot(xnew,ynew)
plt.title('Random Forest Classification')
plt.xlabel('Number of Trees, n')
plt.ylabel('Accuracy, %')
plt.show()

