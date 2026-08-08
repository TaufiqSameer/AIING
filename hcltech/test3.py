import pandas as pd
import re
from collections import defaultdict

train_df = pd.read_csv("train.csv")
from collections import Counter

for course, group in train_df.groupby("Course"):

    print("="*70)
    print(course)

    s3 = Counter()
    s4 = Counter()
    s5 = Counter()
    s6 = Counter()

    for review in group["Reviews"]:

        s = re.split(r'(?<=[.!?])\s+', review)

        while len(s) < 6:
            s.append("")

        s3[s[2]] += 1
        s4[s[3]] += 1
        s5[s[4]] += 1
        s6[s[5]] += 1

    print("Sentence3")
    print(s3)

    print()

    print("Sentence4")
    print(s4)

    print()

    print("Sentence5")
    print(s5)

    print()

    print("Sentence6")
    print(s6)

    break