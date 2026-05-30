import pandas as pd;
import seaborn as sns;
import matplotlib.pyplot as plt;
import numpy as np;

# we are doing analysis on titantic dataset this is best to know basics
# we will how to clean the data
# more good analysis -> better model


# reading dataset using pandas
train = pd.read_csv('Titanic-Dataset.csv')
#we have passengerid,survivded,sibsp,parent child, ticket, fare
# we need to find missing data
print(train.head());
# true -> value is null i,e, NaN
print(train.isnull());

# sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis');
# plt.show();

sns.set_style('whitegrid');
# sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
# plt.show();
# sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
# plt.show();

# sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=40)
# plt.show();

sns.countplot(x='SibSp',data=train);
plt.show();

# data cleaning
sns.boxplot(x='Pclass',y='Age',data=train);
plt.show();

def ch_age(cols):
    Age = cols['Age'];
    Pclass = cols['Pclass'];
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
    
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
    
train['Age'] = train[['Age','Pclass']].apply(ch_age,axis=1)

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis');
plt.show();

train.drop('Cabin',axis=1,inplace=True);
print(train.head());

embark = (pd.get_dummies(train['Embarked'], drop_first=True))
sex = (pd.get_dummies(train['Sex'], drop_first=True))

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True);

train = pd.concat([train,sex,embark],axis=1);

print(train)

train.drop('Survived',axis=1).head();

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'],test_size=0.30,random_state=101);

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression();
logmodel.fit(X_train,Y_train);

predictions = logmodel.predict(X_test);

from sklearn.metrics import confusion_matrix,accuracy_score;

cm = confusion_matrix(Y_test,predictions);
acc = accuracy_score(Y_test,predictions);

print(acc);
