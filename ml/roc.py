import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;
from sklearn.datasets import make_classification    
from sklearn.model_selection import train_test_split;
x,y = make_classification(n_samples=2000,n_classes=2,weights=[1,1],random_state=1);

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1);

from sklearn.metrics import roc_curve;
from sklearn.metrics import roc_auc_score;

from sklearn.ensemble import RandomForestClassifier;

rf_model = RandomForestClassifier();
rf_model.fit(x_train,y_train);

ytain_pref = rf_model.predict(x_train);
print(ytain_pref);

ytest_pred = rf_model.predict_proba(x_test);

from sklearn.linear_model import LogisticRegression;

log_classifer = LogisticRegression();
log_classifer.fit(x_train,y_train);
ytrain_pred = log_classifer.predict_proba(x_train);
ytest_pred = rf_model.predict_proba(x_test);

accuracy_ls = [];