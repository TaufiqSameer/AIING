import numpy as np
import pandas as pd
from io import StringIO

data = '[{"employee_name":"sameer" , "email":"sameertaufq@gmail.com"}]'

print(data);

json_data = '''
[
 {"name":"Sameer","salary":50000,"dept":"IT"},
 {"name":"Rahul","salary":45000,"dept":"HR"},
 {"name":"Anita","salary":60000,"dept":"Finance"},
 {"name":"Priya","salary":55000,"dept":"IT"}
]
'''

df = pd.read_json(StringIO(json_data));

print(df)

df.to_csv("employee.csv");

df2 = df.to_json(orient="records")

print(df2)

