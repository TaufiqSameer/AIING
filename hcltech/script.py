import re
from collections import defaultdict
import numpy as np;
import pandas as pd;
import seaborn as sns;
import matplotlib.pyplot as plt;
from tqdm import tqdm;

train_df = pd.read_csv("train.csv");
test_df = pd.read_csv("test.csv");
def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip()) 

slots = defaultdict(set)

for review in train_df["Reviews"]:
    s = split_sentences(review)

    for i, sentence in enumerate(s):
        slots[i].add(sentence)

for i in sorted(slots):
    print("\n" + "=" * 80)
    print(f"SLOT {i}")

    for x in sorted(slots[i]):
        print(x)
        
lengths = train_df["Reviews"].apply(lambda x: len(split_sentences(x)))
print(lengths.value_counts().sort_index())