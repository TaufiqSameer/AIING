import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;

x = np.linspace(-5.0,5.0,100);
y = np.sqrt(10**2 - x**2);
y = np.hstack([y,-y]);
x = np.hstack([x,-x]);

x1 = np.linspace(-5.0,5.0,100);
y1 = np.sqrt(5**2 - x1**2);
y1 = np.hstack([y1,-y1]);
x1 = np.hstack([x1,-x1]);

plt.scatter(y,x);
plt.scatter(y1,x1);
# plt.show();


df1 = pd.DataFrame(np.vstack([y,x]).T,columns=['X1','X2']);
df1['Y'] = 0;
df2 = pd.DataFrame(np.vstack([y1,x1]).T,columns=['X1','X2']);
df2['Y'] = 1;
df = pd.concat([df1, df2])
print(df.head(3));

X = df[['X1','X2']];
Y = df['Y'];
from sklearn.model_selection import train_test_split;
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=42);

print(y_train);

from sklearn.svm import SVC;
classifier = SVC(kernel='linear');
classifier.fit(x_train,y_train);

from sklearn.metrics import accuracy_score;
y_pred = classifier.predict(x_test);

print(accuracy_score(y_test,y_pred));

df['X1_Square'] = df['X1'] ** 2;
df['X2_Square'] = df['X2'] ** 2;
df['X1*X2'] = (df['X1']  * df['X2']);
print(df.head());

x = df[['X1','X2','X1_Square','X2_Square','X1*X2']];
y = df['Y'];

print(y);
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42);

print(x_train);


import plotly.express as px
fig = px.scatter_3d(df,x='X1',y='X2',z='X1*X2',color='Y');
fig.show();
