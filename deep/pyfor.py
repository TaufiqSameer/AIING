import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyforest import *

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"

df = pd.read_csv(url)

print(df.head())

active_imports()

lst1 = [1,2,3,4,5]
lst2 = [1,4,9,16,25]

plt.plot(lst1,lst2);
plt.xlabel("x-axis");
plt.ylabel("y-axis");
plt.show()

active_imports();