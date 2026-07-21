import numpy as np;
import pandas as pd;
from ydata_profiling import ProfileReport

from sklearn.datasets import load_diabetes;

data_diab = load_diabetes();

df = pd.DataFrame(data=data_diab.data,columns=data_diab.feature_names)
print(df.head())

profile = ProfileReport(df,title="analysis",explorative=True);

profile.to_file("diabetes_report.html")