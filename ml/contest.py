import sweetviz
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

train = pd.read_csv("train.csv");
test = pd.read_csv("train.csv");


print(train.info());
print(test.info());

print(train.head());

report = sweetviz.analyze([train,"Train"],target_feat='SalePrice')

report.show_html("report.html");