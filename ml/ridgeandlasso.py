from sklearn.datasets import fetch_california_housing

df = fetch_california_housing();

# print(df)

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

dataset = pd.DataFrame(df.data);

print(dataset);
print(df.target.shape)

dataset["target"] = df.target;

X = dataset.iloc[:,:-1];
Y = dataset.iloc[:,-1];

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

linreg = LinearRegression();
mse = cross_val_score(linreg,X,Y,scoring="neg_mean_squared_error",cv=5);
mean_mse = np.mean(mse);
print(mean_mse)

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

print(lasso_regressor)
lasso_regressor.fit(X,Y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)
