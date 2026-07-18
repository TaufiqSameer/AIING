import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;
import numpy as np;

df = pd.read_csv("StatewiseTestingDetails.csv");

print(df.head(3));

df["Date"] = pd.to_datetime(df["Date"])

df["Negative"] = pd.to_numeric(df["Negative"], errors="coerce")
df["Positive"] = pd.to_numeric(df["Positive"], errors="coerce")

print(df.info());


print(df.isnull().sum());

print("Number of unqiue states " , df['State'].nunique());

print("Span " , df["Date"].min() , " to " , df["Date"].max());

print(df.groupby("State").size());

state = "Telangana";

temp = df[df['State'] == state];

print(temp);

plt.figure(figsize=(10,6));
plt.plot(temp["Date"],temp["TotalSamples"]);
plt.show();
