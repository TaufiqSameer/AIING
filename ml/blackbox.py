import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("Churn_Modelling.csv")
X = df.iloc[:,3:13]
y = df.iloc[:,13]


print(df.head())
print(df.isnull().sum())
# Overfitting:
# Model is a blackbox. Thought not enirely we get some of input and then some output in short we get training and testing dataset 
# we validate using test dataset we use different performance metrics to evaluate our model more the metrics better the model
# in train dataset and it should be similar to the train dataset 
# lets have a case where we get good score with training but not with training set at that time it is said to be overfitting
# in short, we make model on the contraints imposed on the train dataset i,e made specifically for the train constraints
# Underfitting:
# we get low accuracy on both the training and testing dataset 
# data leakage: model is actually train and it should know about the train data not about test data and if it knows then obviously it will perform
# better on test data 
#
# API: 
#
#
#
#
#
#
#
#
#
#
#

geography = pd.get_dummies(X["Geography"],drop_first=True)
print(geography)
gender = pd.get_dummies(X["Gender"],drop_first=True)
print(gender)

X = pd.concat([X,geography,geography],axis=1)
X = X.drop(['Geography',"Gender"],axis=1)

from sklearn.model_selection import train_test_split;
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train,y_train)

import pickle
pickle.dump(clf,open("classifier.pkl","wb"))

import lime 
from lime import lime_tabular

interpretor = lime_tabular.LimeTabularExplainer(training_data=np.array(X_train),feature_names=X_train.columns,mode="classification")

exp = interpretor.explain_instance(data_row=X_test.iloc[4],predict_fn=clf.predict_proba)

exp.save_to_file("lime_explanation.html")
