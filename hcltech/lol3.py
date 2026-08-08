import pandas as pd
import re
import matplotlib.pyplot as plt;
import seaborn as sns;

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
import re

# for old, new in zip(old_predictions, predictions):
#     if old == new:
#         same += 1

# print(f"{same}/{len(predictions)} unchanged")
import re

def anonymize_first_sentence(review, course):
    sentences = re.split(r'(?<=[.!?])\s+', review)
    if sentences:
        pattern = re.escape(course)
        sentences[0] = re.sub(pattern,"this course",
            sentences[0],
            flags=re.IGNORECASE,
        )

    return " ".join(sentences)

train_df["Reviews"] = [
    anonymize_first_sentence(review, course)
    for review, course in zip(train_df["Reviews"], train_df["Course"])
]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,3),
    binary=True,
    sublinear_tf=True,
    min_df=2,
    max_df=0.9,
)

X_train = tfidf.fit_transform(train_df["Reviews"])
X_test = tfidf.transform(test_df["Reviews"])

nn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=10
)

nn.fit(X_train)
distances, indices = nn.kneighbors( X_test, n_neighbors=20)

for q in range(10):     
    print("="*80)
    print("TEST REVIEW")
    print(test_df.iloc[q]["Reviews"])
    print()

    sims = 1 - distances[q]

    for rank in range(20):
        idx = indices[q][rank]

        print(
            f"Rank {rank+1:2d}",
            f"Sim={sims[rank]:.4f}",
            f"TrainIndex={train_df.iloc[idx]['Index']}"
        )

submission = pd.DataFrame({
    "Index": test_df["Index"],
    "Index_list": [
        str(list(train_df.iloc[idx[:10]]["Index"]))
        for idx in indices
    ]
})
submission.to_csv("submissionll_critical.csv", index=False)

q = 0

print("QUERY")
print(test_df.iloc[q]["Reviews"])

print("\nTOP 10\n")

for rank, idx in enumerate(indices[q][:10], 1):
    print("="*80)
    print(f"Rank {rank}")
    print(f"Similarity: {1 - distances[q][rank-1]:.4f}")
    print()
    print(train_df.iloc[idx]["Reviews"])