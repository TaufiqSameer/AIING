import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;

train_df = pd.read_csv("train.csv");
test_df = pd.read_csv("test.csv");

import pandas as pd
import re
from collections import Counter

train = pd.read_csv("train.csv")

def split(review):
    s = re.split(r'(?<=[.!?])\s+', review.strip())
    while len(s) < 6:
        s.append("")
    return s[:6]

for pos in range(2, 6):

    cnt = Counter()

    for _, group in train.groupby("Course"):

        for review in group["Reviews"]:
            cnt[split(review)[pos]] += 1

    print("="*60)
    print(f"Sentence {pos+1}")
    print(cnt.most_common(30))
    
import pandas as pd
import re

train = pd.read_csv("train.csv")

def split(review):
    s = re.split(r'(?<=[.!?])\s+', review.strip())
    while len(s) < 6:
        s.append("")
    return s[:6]

for course, group in train.groupby("Course"):

    combos = set()

    for review in group["Reviews"]:

        s = split(review)

        combos.add(
            (
                s[2],
                s[3],
                s[4],
                s[5]
            )
        )

    print(
        f"{course:45}",
        len(combos),
        "/",
        len(group)
    )
    
import pandas as pd
import re

train = pd.read_csv("train.csv")

def split(review):
    s = re.split(r'(?<=[.!?])\s+', review.strip())
    while len(s) < 6:
        s.append("")
    return s[:6]

stats = []

for course, group in train.groupby("Course"):

    rows = [split(r) for r in group["Reviews"]]

    for pos in range(6):

        unique = len(set(r[pos] for r in rows))

        stats.append({
            "Course": course,
            "Sentence": pos+1,
            "Unique": unique
        })

stats = pd.DataFrame(stats)

print(
    stats.groupby("Sentence")["Unique"].describe()
)

import pandas as pd
import re
from collections import Counter

train = pd.read_csv("train.csv")

def split(review):
    s = re.split(r'(?<=[.!?])\s+', review.strip())
    while len(s) < 6:
        s.append("")
    return s[:6]

counter = Counter()

for _, group in train.groupby("Course"):

    reviews = [split(r) for r in group["Reviews"]]

    n = len(reviews)

    for i in range(n):

        for j in range(i+1, n):

            diff = []

            for k in range(2,6):

                if reviews[i][k] != reviews[j][k]:
                    diff.append(k+1)

            counter[tuple(diff)] += 1

print(counter.most_common(50))

import pandas as pd
import re
from collections import Counter

train = pd.read_csv("train.csv")

def split(review):
    s = re.split(r'(?<=[.!?])\s+', review.strip())
    while len(s) < 6:
        s.append("")
    return s[:6]

for course, group in train.groupby("Course"):

    print("="*80)
    print(course)

    reviews = [split(r) for r in group["Reviews"]]

    cnt = Counter()

    for review in reviews:

        cnt[
            (
                review[2],
                review[3],
                review[4],
                review[5]
            )
        ] += 1

    print("Unique patterns :", len(cnt))
    print("Largest cluster :", cnt.most_common(10))