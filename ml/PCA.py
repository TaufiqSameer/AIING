import matplotlib.pyplot as plt;
import pandas as pd;
import numpy as np;
import seaborn as sns;

from sklearn.datasets import load_breast_cancer;

cancer = load_breast_cancer();

# print(cancer.keys());

# print(cancer['DESCR']);

df = pd.DataFrame(cancer['data'],columns=cancer['feature_names']);

print(df);

# dimension reduction LoL xD.45 Critical Ops
from sklearn.preprocessing import StandardScaler;


scalar = StandardScaler();
scalar.fit(df);
scaled_data = scalar.transform(df);

from sklearn.decomposition import PCA;

pca = PCA(n_components=2);

pca.fit(scaled_data);

x_pca = pca.transform(scaled_data);

print(scaled_data.shape)

plt.figure(figsize=(8,6));
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma');
plt.xlabel("First Principal Componenet")
plt.ylabel("Second Principal Componenet")

plt.show();
