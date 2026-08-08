import re
from collections import defaultdict
import numpy as np;
import pandas as pd;
import seaborn as sns;
import matplotlib.pyplot as plt;
from tqdm import tqdm;

train_df = pd.read_csv("train.csv");
test_df = pd.read_csv("test.csv");

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

word_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 3),
    binary=True,
    sublinear_tf=True,
    min_df=2,
    max_df=0.9,
)

X_train_word = word_vectorizer.fit_transform(train_df["Reviews"])
X_test_word = word_vectorizer.transform(test_df["Reviews"])


print("Comp1")

char_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,3),
    binary=True,
    sublinear_tf=True,
    min_df=2,
    max_df=0.9,
)

X_train_char = char_vectorizer.fit_transform(train_df["Reviews"])
X_test_char = char_vectorizer.transform(test_df["Reviews"])

print("Comp2")
from scipy.sparse import hstack

X_train = hstack([X_train_word, X_train_char])
X_test = hstack([X_test_word, X_test_char])

import time

nn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=10,
)

start = time.time()
print("Comp3")
nn.fit(X_train)
distances, indices = nn.kneighbors(X_test, n_neighbors=100)

unique_course_counts = []

for row in indices:
    courses = train_df.iloc[row]["Course"]
    unique_course_counts.append(courses.nunique())

import pandas as pd

print(pd.Series(unique_course_counts).describe())

print(pd.Series(unique_course_counts).value_counts().sort_index())

import matplotlib.pyplot as plt
import numpy as np

top1 = distances[:,0]
top2 = distances[:,1]
top10 = distances[:,9]

plt.figure(figsize=(8,5))
plt.hist(top1, bins=100, alpha=0.6, label="Top1")
plt.hist(top2, bins=100, alpha=0.6, label="Top2")
plt.hist(top10, bins=100, alpha=0.6, label="Top10")
plt.legend()
plt.xlabel("Cosine Distance")
plt.ylabel("Count")
plt.show()

gap = distances[:,1] - distances[:,0]

plt.figure(figsize=(8,5))
plt.hist(gap, bins=100)
plt.xlabel("Top2 - Top1 distance")
plt.ylabel("Queries")
plt.show()

for i in np.argsort(top1)[-20:]:
    print("="*80)
    print("TEST REVIEW")
    print(test_df.loc[i, "Reviews"])
    print()

    print("BEST MATCH")
    idx = indices[i,0]
    print(train_df.iloc[idx]["Reviews"])

    print()
    print("Distance:", distances[i,0])
    
spread = distances[:,9] - distances[:,0]

plt.hist(spread, bins=100)
plt.xlabel("Top10 - Top1")
plt.show()