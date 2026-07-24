import seaborn as sns;
print(sns.get_dataset_names());

df = sns.load_dataset('iris');
print(df.head());

from pandas_visual_analysis import VisualAnalysis

vis = VisualAnalysis(df)
print(dir(vis))


import matplotlib.pyplot as plt;

plt.scatter(df['sepal_length'],df['petal_width']);
plt.show();