import scipy.stats as stats;

import seaborn as sns;
import pandas as pd;
import numpy as np;

dataset = sns.load_dataset('tips');

print(dataset.head());

dataset_table = pd.crosstab(dataset['sex'],dataset['smoker']);

print(dataset_table);

observed = dataset_table.values;

print(observed);

val = stats.chi2_contingency(dataset_table);

print(val)