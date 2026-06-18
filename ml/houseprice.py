import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;

pd.pandas.set_option('display.max_columns',None);

dataset = pd.read_csv('train.csv');

# print(dataset.shape);

# print(dataset.head());

lol = [features for features in dataset.columns if dataset[features].isnull().sum() >1];
# for f in lol:
#     print(f, np.round(dataset[f].isnull().mean(), 4), 'missing values')
    
for f in lol:
    data = dataset.copy();
    
    data[f] = np.where(data[f].isnull(),1,0);
    
    data.groupby(f)['SalePrice'].median().plot.bar()
    plt.title(f);
    # plt.show();


numerical_features = [f for f in dataset.columns if dataset[f].dtype != 'O'];

# print(len(numerical_features));


# for col in dataset.columns:
#     print(col)
# for feature in numerical_features:
#     data = dataset.copy();
    
#     if 0 in data[feature].unique():
#         pass
#     else:
#         data[feature] = np.log(data[feature]);
#         data['SalePrice'] = np.log(data['SalePrice']);
#         plt.scatter(data[feature],data['SalePrice']);
#         plt.xlabel("feature");
#         plt.ylabel("SalePrice");
#         plt.title('Plotting');
#         # plt.show();
        
        
from sklearn.model_selection import train_test_split;
X = dataset.drop('SalePrice', axis=1)
y = dataset['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=0
)
# print(x_train);


num_features = ['LotFrontage','LotArea','1stFlrSF','SalePrice'];

for f in num_features:
    dataset[f] = np.log(dataset[f]);


print(dataset.head());

 
from sklearn.preprocessing import MinMaxScaler;

feature_scale = [f for f in dataset.columns if f not in ['Id', 'SalePrice']]
scalar = MinMaxScaler();
scalar.fit(dataset[feature_scale]);


from sklearn.linear_model import Lasso;
from sklearn.feature_selection import SelectFromModel;

d2 = pd.read_csv('train.csv');
print(d2.head());

X_train = d2.drop(['Id','SalePrice'],axis = 1);

feature = SelectFromModel(Lasso(alpha=0.05,random_state=0));
feature.fit(X_train,y_train);

print(feature.get_support());
        