import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.feature_selection import SelectKBest;
from sklearn.feature_selection import chi2;


data = pd.read_csv("train2.csv");
print(data.head());
x = data.iloc[:,0:20];
y = data.iloc[:,-1];

best_features = SelectKBest(score_func=chi2,k=10);
fit = best_features.fit(x,y);

dfscores = pd.DataFrame(fit.scores_);
dfcolumns = pd.DataFrame(x.columns);

featurescores = pd.concat([dfcolumns,dfscores],axis=1);
featurescores.columns = ['Specs','Score'];

print(featurescores.nlargest(10,'Score'));

from sklearn.ensemble import ExtraTreesClassifier;
model = ExtraTreesClassifier();
model.fit(x,y);

print(model.feature_importances_);

feat_imp = pd.Series(model.feature_importances_,index = x.columns);

feat_imp.nlargest(10).plot(kind='area');
plt.show();

import seaborn as sns;
corrmat = data.corr();

top_corr = corrmat.index();
