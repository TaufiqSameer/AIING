import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;

df = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv');

# print(df.head(5));

### Univariate analysis

print(df.shape)

setosa = (df.loc[df['species'] == 'setosa'])
virginica = (df.loc[df['species'] == 'virginica']);
versicolor = (df.loc[df['species'] == 'versicolor']);


plt.plot(setosa['sepal_length'],np.zeros_like(setosa['sepal_length']),'o')
plt.plot(virginica['sepal_length'],np.zeros_like(virginica['sepal_length']),'o')
plt.plot(versicolor['sepal_length'],np.zeros_like(versicolor['sepal_length']),'o')
plt.xlabel("Petals_length")
plt.show();


##bivariate

sns.FacetGrid(df,hue="species",height=5).map(plt.scatter,"sepal_length","sepal_width").add_legend();
plt.show();

sns.pairplot(df,hue='species',height=5);
plt.show();