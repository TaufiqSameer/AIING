import pandas as pd;
import numpy as np;
import seaborn as sns;

df = pd.read_csv("winequality-white.csv",sep=";");
# print(df.head());


print(df['quality'].unique());

X = df.iloc[:,:-1];
y = df.iloc[:,-1];

print(y.unique);

from time import time;
from sklearn.model_selection import RepeatedStratifiedKFold,cross_val_score

from  sklearn.ensemble import RandomForestClassifier;

staert = time();

model = RandomForestClassifier(n_estimators=100,n_jobs=10)

cv = RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=4);


n_scores = cross_val_score(model, X, y, scoring='accuracy')
end = time()
print(end-staert);
