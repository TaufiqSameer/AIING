from collections import Counter
import re
import pandas as pd
train_df = pd.read_csv("train.csv");
test_df = pd.read_csv("test.csv");

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

train_counter = Counter()
test_counter = Counter()

# for review in train_df["Reviews"]:
#     train_counter.update(tokenize(review))

# for review in test_df["Reviews"]:
#     test_counter.update(tokenize(review))

rows = []

# for word in train_counter:
#     rows.append({
#         "word": word,
#         "train": train_counter[word],
#         "test": test_counter[word],
#         "ratio": (train_counter[word]+1)/(test_counter[word]+1)
#     })

import pandas as pd

# df = pd.DataFrame(rows)

# df = df.sort_values("ratio", ascending=False)

# print(df.head(100))                             l,
train_df = pd.read_csv("train.csv")
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Pick one test review
i = 0

sim = cosine_similarity(X_test[i], X_train).ravel()

top = np.argsort(-sim)[:20]

for rank, idx in enumerate(top, 1):
    print(rank,
          f"{sim[idx]:.6f}",
          train_df.iloc[idx]["Course"],
          train_df.iloc[idx]["Index"])