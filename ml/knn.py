import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("Classified data",index_col=0);

print(df.head(3));

from sklearn.preprocessing import StandardScaler;

scalar = StandardScaler();

scalar.fit(df.drop('TARGET CLASS',axis=1));

scaled = scalar.transform(df.drop('TARGET CLASS',axis=1));

dff = pd.DataFrame(scaled,columns=df.columns[:-1]);
print(dff.head());

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(scaled,df['TARGET CLASS'],test_size=0.30);

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1);


knn.fit(x_train,y_train);

pred = knn.predict(x_test);

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred));

error_rate = [];

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i);
    knn.fit(x_train,y_train);
    pred_i = knn.predict(x_test);
    error_rate.append(np.mean(pred_i != y_test));
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


knn = KNeighborsClassifier(n_neighbors=23)

knn.fit(x_train,y_train)
pred = knn.predict(x_test)

print('WITH K=23')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))



