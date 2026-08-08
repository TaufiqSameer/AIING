import re
from collections import Counter

def first_sentence(text):
    s = re.split(r'(?<=[.!?])\s+', text)
    return s[0]

import pandas as pd
train_df = pd.read_csv("train.csv");
test_df = pd.read_csv("test.csv");
train_first = Counter(train_df["Reviews"].apply(first_sentence))
test_first = Counter(test_df["Reviews"].apply(first_sentence))

print("Unique train openings:", len(train_first))
print("Unique test openings:", len(test_first))

print("\nMost common TRAIN")
print(train_first.most_common(30))

print("\nMost common TEST")
print(test_first.most_common(30))