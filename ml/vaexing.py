import numpy as np;
import pandas as pd;
import seaborn as sns;
import matplotlib.pyplot as plt;
import time;

print(sns.get_dataset_names());

df = sns.load_dataset("planets");
print(df.head());

import dtale;
d = dtale.show(df,host="127.0.0.1")
print(d._main_url)

while True:
    time.sleep(1)
