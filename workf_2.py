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
dataset = pd.read_csv(r"C:\Users\Abdul Basit Aftab\Desktop\MachineLearningThesis\Data + Codes\data_set_training.csv") #training data set
dataset2 = pd.read_csv(r"C:\Users\Abdul Basit Aftab\Desktop\MachineLearningThesis\Data + Codes\data_m5.CSV") # testing data set
dataset2.fillna(0) # fiiling 0 on the null values
dataset['avg'] = dataset.iloc[:,4:7].mean(axis=1)
dataset2['avg'] = dataset2.iloc[:,4:7].mean(axis=1)
dataset['rollingaverage'] = dataset.avg.rolling(3).mean()
dataset['rollingaverage'] = dataset.rollingaverage.fillna(0)
dataset2['rollingaverage'] = dataset2.avg.rolling(3).mean()
dataset2['rollingaverage'] = dataset2.rollingaverage.fillna(0)
#dataset2=dataset2.iloc[[dataset2['avg']>=5 &  dataset2['avg']<=10]]
mask = ((dataset['rollingaverage'] > 5) & (dataset2['rollingaverage'] <= 10))
dataset = dataset.loc[mask]
mask = ((dataset2['avg'] > 5) & (dataset2['avg'] <= 10))
dataset2 = dataset2.loc[mask]
X_train = dataset.iloc[:,4:7].values
#acb = dataset2.iloc[:,-1]
#X_train = dataset.iloc[:,4:7].values
X_train = dataset.iloc[:,4:7].values
X_train_new = dataset.iloc[:,7] #quality





#%% TRaining Testing Dataset
Y_train = dataset.iloc[49:,7].values # getting quality values
Y_train = 100 - Y_train


X_train = dataset.iloc[49:,-1].values

#X_train = dataset.iloc[:,4:7].values

#X_train = X_train.append(dataset['Vibration']).values
#X_train = dataset.iloc[:,12].values
#X_train= X_train.reshape(-3, 3)
X= X_train
#Y_train= Y_train.reshape(-1, 1)
y= Y_train
#aas=X.mean(axis = 1)
#plt.scatter(X[:,1],y) # ploting graph of product quality

#plt.scatter(X[:,0:1],Y_train) # plotting of die degradation %

plt.scatter(dataset.iloc[49:,-1],Y_train) # plotting of die degradation %
#plt.scatter(X_test_new[:,1],Y_test_new)
plt.ylim(0,100)
plt.title('Current Verses Die Degeradation')
plt.xlabel('Current Ratings , A')
plt.ylabel('Die Degradation , %')

#plt.legend(['Training Dataset','Testing Dataset'],loc=2)
plt.show()
#%% Test Train Split
X_train_new , X_test_new , Y_train_new , Y_test_new = train_test_split(X,y,test_size= 0.3, random_state= 0)
X_train= X_train.reshape(-1, 1)

X= X_train

Y_train= Y_train.reshape(-1, 1)
y= Y_train

#%% Data Preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

Y_train = Y_train.reshape(-1,1)
scaledX = scaler.fit_transform(X_train)
scaledY = scaler.fit_transform(Y_train)

#%% For accuract score
X_test_newest = dataset2.iloc[:,-1].values
#Y_train_newest = dataset.iloc[:83,7].values
X_test_newest= X_test_newest.reshape(-1, 1)
X_test_new= X_test_new.reshape(-1, 1)
Y_test_new=Y_test_new.reshape(-1,1)
#X_train_newest= X_train_newest.reshape(-1, 1)
#Y_train_newest= Y_train_newest.reshape(-1, 1)
X_train_new= X_train_new.reshape(-1, 1)

#%%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_new = sc.fit_transform(X_train_new)
X_test_newest = sc.transform(X_test_newest)
X_test_new = sc.transform(X_test_new)

# Fitting Logistic Regression to the Training set
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state = 0)
#classifier.fit(X_train_new, Y_train_new)
#%% Logistic Regression


from sklearn.linear_model import LogisticRegression
#lm = LogisticRegression(random_state=0)
mul_lr = LogisticRegression(multi_class='multinomial', solver='newton-cg')
mul_lr.fit(X_train_new, Y_train_new)
#X_train_new= X_train_new.reshape(-1, 1)
#Y_train_new= Y_train_new.reshape(-1, 1)
#
#X_test_new= X_test_new.reshape(-1, 1)
#lm.fit(X_train_new,Y_train_new)
#
#
y_pred_lm_split = mul_lr.predict(X_test_new)
thisisaccuracy_lm = accuracy_score(Y_test_new, y_pred_lm_split)
##thisisaccuracy_lm = accuracy_score(Y_test_new, y_pred_lm_split)
print(y_pred_lm_split)
print(thisisaccuracy_lm)
#lm_conf_mat = confusion_matrix(Y_test_new,y_pred_lm_split)
#print(lm_conf_mat)

#%% Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
clf2 = tree.DecisionTreeClassifier(criterion='entropy')
#clf2 = tree.DecisionTreeClassifier(max_depth=20)
#X_test_new = sc.inverse_transform(X_test_new,copy=None)
#X_train_new = sc.inverse_transform(X_train_new,copy=None)
clf2 = clf2.fit(X_train_new,Y_train_new)
y_pred_clf2_split = clf2.predict(X_test_new)
thisisaccuracy_clf2 = accuracy_score(Y_test_new, y_pred_clf2_split)
#
#
print(thisisaccuracy_clf2)
clf2_conf_mat = confusion_matrix(Y_test_new,y_pred_clf2_split)
print(clf2_conf_mat)

#%% Decision tree Trial
import numpy as np
a =np.array([])
b=[]
iter_x_values = np.arange(1,1000)


for i in iter_x_values:
    clf2 = tree.DecisionTreeClassifier(max_depth=i,criterion='entropy')
    clf2 = clf2.fit(X_train_new,Y_train_new)
    y_pred_clf2_split = clf2.predict(X_test_new)
    thisisaccuracy_clf2 = accuracy_score(Y_test_new, y_pred_clf2_split)
    b.append(thisisaccuracy_clf2)
    print(i)
    print(thisisaccuracy_clf2)

#%%Random Forest Classifier %90.8
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(criterion='entropy',min_samples_leaf=1,min_samples_split=2,n_estimators=18,random_state=0,bootstrap=True)
#random forest is criterion='entropy', Min_samples_leaf=1, min_samples_split=2, n_estimators=10, n_job=2, oob_score=False, random_state=1, warm_start=False.
rfc.fit(X_train_new,Y_train_new)
aaa=rfc.predict_proba(X_train_new)

y_pred_rfc = rfc.predict(X_test_new)
thisisaccuracy_rfc = accuracy_score(Y_test_new, y_pred_rfc)
print(thisisaccuracy_rfc)
#
#
#
#rfc_conf_mat = confusion_matrix(Y_test_new,y_pred_rfc)
#print(rfc_conf_mat)
#%%
print(y_pred_lm_split)
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
nb = GaussianNB()
from sklearn.naive_bayes import BernoulliNB
bn = BernoulliNB()
nb.fit(X_train_new,Y_train_new)
#bn.fit(X_train_new,Y_train_new)
#a=nb.predict(X_test_new)
y_pred_nb = nb.predict(X_test_new)


thisisaccuracy_nb = accuracy_score(Y_test_new, y_pred_nb)
print(thisisaccuracy_nb)
#
#clf4_conf_mat = confusion_matrix(Y_test_new,y_pred_nb)
#print(clf4_conf_mat)




#%%PLOTTING Naive Bayes

X_test_new = sc.inverse_transform(X_test_new,copy=None)
plt.scatter(X_test_new[:,],Y_test_new)
plt.scatter(X_test_new[:,],y_pred_nb)
plt.title('Naive Bayes Algorithm')
plt.xlabel('Current Ratings , A')
plt.ylabel('Die Degradation , %')
plt.legend(['Training Dataset','Testing Dataset'],loc=2)
plt.show()

#%%
plt.scatter(X_test_new[:,],a)
plt.scatter(X_test_new[:,],Y_test_new)
plt.title('Naive Bayes Algorithm')
plt.xlabel('Current Ratings , A')
plt.ylabel('Product quality , %')
plt.legend(['Training Dataset','Testing Dataset'],loc=2)
plt.show()
#%% PLOTTING LOGISTIC REGRESSIOn
#X_test_new = sc.inverse_transform(X_test_new,copy=None)
plt.scatter(X_test_new,Y_test_new)
plt.scatter(X_test_new,y_pred_lm_split)
plt.title('Logistic Regression')
plt.xlabel('Current Ratings , A')
plt.ylabel('Die Degradation , %')
plt.legend(['Training Dataset','Testing Dataset'],loc=2)
plt.show()

#%%PLOTTING RANDOM FOREST
#X_test_new = sc.inverse_transform(X_test_new,copy=None)
plt.scatter(X_test_new,Y_test_new)
plt.scatter(X_test_new,y_pred_rfc)
plt.title('Random Forest Classification')
plt.xlabel('Current Ratings , A')
plt.ylabel('Die Degradation , %')
plt.legend(['Training Dataset','Testing Dataset'],loc=2)
plt.show()
#%%PLOTTING DECISION TREE
#X_test_new = sc.inverse_transform(X_test_new,copy=None)
plt.scatter(X_test_new,Y_test_new)
plt.scatter(X_test_new,y_pred_clf2_split)
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
#X_train_new = sc.inverse_transform(X_train_new,copy=None)
#X_test_new = sc.inverse_transform(X_test_new,copy=None)
fig = plt.figure(figsize=(125,100))
_ = tree.plot_tree(clf2, 
                   feature_names=X_train_new,  
                   class_names=['10','20','30','40','50','60','70','80','90','100'],
                   filled=True)



#%%
import numpy as np
a =np.array([])
b=[]
iter_x_values = np.arange(1,120)


for i in iter_x_values:
    rfc2 =RandomForestClassifier(criterion='entropy',min_samples_leaf=1,min_samples_split=2,n_estimators=i,random_state=0,bootstrap=True) 
    #RandomForestClassifier(n_estimators=i,random_state=0,bootstrap=True)
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

