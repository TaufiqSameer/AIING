import pandas as pd;
import numpy as np;

df = pd.read_csv("BankNote_Authentication.csv");

print(df.head())
y = df.iloc[:,-1];
x = df.iloc[:,:-1];

print(x.head())

from sklearn.model_selection import train_test_split;

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
from sklearn.ensemble import RandomForestClassifier;

classifier = RandomForestClassifier(n_estimators=30);
classifier.fit(x_train,y_train);

y_pred = classifier.predict(x_test);

from sklearn.metrics import accuracy_score;

score = accuracy_score(y_test,y_pred);

print(score);

import pickle;

pickle_out = open("classifier.pkl","wb");
pickle.dump(classifier,pickle_out);
pickle_out.close();

classifier.predict([2,3,4,1]);




