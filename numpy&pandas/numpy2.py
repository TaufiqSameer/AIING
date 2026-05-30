import numpy as np
import pandas as pd
from io import StringIO,BytesIO

df = pd.read_csv('mercedesbenz.csv') # loading the csv using pandas

print(df.head(2)) # to get first two rows of the csv file
print(df.isnull().sum()) # to check null values
print(df.info()) # how many rows,columns dtypes and memory usage
print(df.describe()) # Gives count median std and the central tendenices 

test1_df = pd.read_csv('test1.csv')
print(test1_df);

csv_data = """name,col1,col2,col3,col4
row1,0,1,2,3
row2,4,5,6,7
row3,8,9,10,11
row4,12,13,14,15
row5,16,17,18,19
"""

print(type(csv_data));

df2 = pd.read_csv(StringIO(csv_data),usecols=lambda x : x.lower() in ['col1','col2'])
df3 = pd.read_csv(StringIO(csv_data),dtype=object)



# print(df2.loc([0]['col1']))
print(df3);

df4 = pd.read_csv(StringIO(csv_data),usecols=['col1','col2'],index_col=False)
print(df4)