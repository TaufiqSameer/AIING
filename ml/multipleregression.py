import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

dataset = pd.read_csv("startup.csv");

# print(dataset.head(3));

X = dataset.iloc[:,:-1];
Y = dataset.iloc[:,-1];

# print(X.head());
# print(Y.head());

states = pd.get_dummies(dataset['State'],drop_first=True);

# print(states);

X = X.drop('State', axis = 1);
X = pd.concat([X,states],axis=1);

# print(X.head(2));

from sklearn.model_selection import train_test_split;
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

# print(x_train)
# print()
# print(x_test)
# print()
# print(y_test)
# print()
# print(y_train)
# print()

from sklearn.linear_model import LinearRegression;

linear = LinearRegression();
linear.fit(x_train,y_train);

pred = linear.predict(x_test);

print(pred);

from sklearn.metrics import r2_score;


score = r2_score(y_test,pred)


print(score);

