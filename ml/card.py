import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;

import sklearn;
import scipy;
from imblearn.under_sampling import NearMiss
import pylab
data = pd.read_csv("creditcard.csv",sep=',');

# print(data.head(3));

print(data.info());

column = data.columns.tolist();

columns = [c for c in column if c not in ["Class"]];

target = "Class";

state = np.random.RandomState(42);

X = data[columns];
y = data[target];

print(X.shape);
print(y.shape);

print(data.isnull().values.any());

count_classes = data['Class'].value_counts(sort=True)

count_classes.plot(kind = 'bar' , rot = 0);

plt.title("Transaction Class distribution");
plt.xticks(range(2), ['Not Fraud', 'Fraud'])

plt.xlabel("Class");

plt.ylabel("Frequency");
plt.show();

fraud = data[data['Class'] == 1];
normal = data[data['Class'] == 0];

print(fraud.shape,normal.shape);

nm = NearMiss();
x_res, y_res = nm.fit_resample(X, y)
print(x_res,y_res);

from collections import Counter;

print("Orginial set" ,Counter(y))
print("resample " , Counter(y_res));


from imblearn.combine import SMOTETomek;
from imblearn.under_sampling import NearMiss

smk = SMOTETomek(random_state=42);
x_res,y_res = smk.fit_resample(X,y);

print(x_res.shape, y_res.shape);






