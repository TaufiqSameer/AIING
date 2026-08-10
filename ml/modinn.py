import pandas as pd;
from time import time
import os
os.environ["MODIN_ENGINE"] = "dask"

st = time();

pd.read_csv("train.csv")

ed = time()

print(ed-st)

start = time()
import modin.pandas as pm
pm.read_csv("train.csv")
end = time()

print(end -start)
