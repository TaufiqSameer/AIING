import numpy as np;
import pandas as pd;
from prophet import Prophet
import matplotlib.pyplot as plt;

df = pd.read_csv("airline_passengers.csv", skipfooter=1, engine="python")

print(df.head());

df.columns = ["ds", "y"]

df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")

df = df.dropna()

print(df.head());


print(df.tail());

print(df.head());

df['ds'] = pd.to_datetime(df['ds']);

print(df.head());
print(df.info())
print(df.tail())
print(df.dtypes)
print(df.shape)


model = Prophet();

print(df.columns);

df.columns=['ds','y'];

print(df.head());

model.fit(df);


future_dates = model.make_future_dataframe(periods=365)

print(model.component_modes);

prediction = model.predict(future_dates);

print(prediction.head());