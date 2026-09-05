import pandas as pd 
df = pd.read_csv('titanic.csv',usecols=['Pclass','Age','Fare','Survived'])
print(df.head())

df['Age'].fillna(df.Age.median(),inplace=True)

print(df.isnull().sum())

X = df.iloc[:,1:]
y = df.iloc[:,0]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)

print(X_train)
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)

print(X_train_scaled)
from sklearn.linear_model import LogisticRegression
classify = LogisticRegression()
classify.fit(X_train_scaled,y_train)
classify.predict(X_train_scaled)

