import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;

pd.pandas.set_option('display.max_columns',None);

dataset = pd.read_csv('train.csv');

print(dataset.shape);

print(dataset.head());

lol = [features for features in dataset.columns if dataset[features].isnull().sum() >1];
for f in lol:
    print(f, np.round(dataset[f].isnull().mean(), 4), 'missing values')
    
for f in lol:
    data = dataset.copy();
    
    data[f] = np.where(data[f].isnull(),1,0);
    
    data.groupby(f)['SalePrice'].median().plot.bar()
    plt.title(f);
    plt.show();


numerical_features = [f for f in dataset.columns if dataset[f].dtype != 'O'];

print(len(numerical_features));
