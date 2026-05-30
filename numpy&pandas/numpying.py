import numpy as np
import pandas as pd

l1 = [1,2,3,4,5]
l2 = [10,11,12,14,15]
l3 = [10,11,12,14,15]


# print(l2 + l3)
# print(l2.shape)

mynew = np.array([l1,l2,l3]);
print(mynew)
print(mynew.shape)
mynew = mynew.reshape(5,3)
# l2.reshape(2,3)
print(mynew)
print(mynew.shape)
# print(l2)

print(mynew[1][0])

print(mynew[:1,0:2])

arr = np.arange(0,10);
print(arr)

print(np.linspace(1,10,50))

arr[3:] = 50

print(arr)

print(arr[arr<2])

print(np.ones((10,10),dtype=int))

########## pandas ################

### what are data frames

'''

'''

df = pd.DataFrame(np.arange(0,20).reshape(5,4),index=['row1','row2','row3','row4','row5'],columns=['col1','col2','col3','col4']);

print(df.head());

print(df.to_csv("test1.txt"))

print(df.loc['row1'])
print(df.iloc[0:2,0:1])
print(df.iloc[0:2,0:1].values.shape)
print(df['col1'].value_counts())

# unique to get unique values 
