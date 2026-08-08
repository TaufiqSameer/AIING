import pandas as pd
import re
import matplotlib.pyplot as plt;
import seaborn as sns;

import re

def anonymize_first_sentence(review, course):
    sentences = re.split(r'(?<=[.!?])\s+', review)
    if sentences:
        pattern = re.escape(course)
        sentences[0] = re.sub(
            pattern,
            "this course",
            sentences[0],
            flags=re.IGNORECASE,
        )
    return " ".join(sentences)

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df["Reviews"] = [
    anonymize_first_sentence(review, course)
    for review, course in zip(train_df["Reviews"], train_df["Course"])
]

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf12 = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2),
    binary=True,
    sublinear_tf=True,
    min_df=2,
    max_df=0.9,
)

tfidf13 = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,3),
    binary=True,
    sublinear_tf=True,
    min_df=2,
    max_df=0.9,
)

X12_train = tfidf12.fit_transform(train_df["Reviews"])
X12_test = tfidf12.transform(test_df["Reviews"])

X13_train = tfidf13.fit_transform(train_df["Reviews"])
X13_test = tfidf13.transform(test_df["Reviews"])

from sklearn.metrics.pairwise import cosine_similarity

batch_size = 200

all_indices = []

import numpy as np
from tqdm import tqdm
for start in tqdm(range(0, X12_test.shape[0], batch_size)):
    end = min(start + batch_size, X12_test.shape[0])

    sim12 = cosine_similarity(X12_test[start:end], X12_train)
    sim13 = cosine_similarity(X13_test[start:end], X13_train)

    sim = 0.7 * sim13 + 0.3 * sim12

    indices = np.argsort(-sim, axis=1)[:, :10]

    all_indices.extend(indices)


indices = np.array(all_indices)

submission = pd.DataFrame({
    "Index": test_df["Index"],
    "Index_list": [
        str(list(train_df.iloc[idx]["Index"]))
        for idx in indices
    ]
})

submission.to_csv("submission_ensemble.csv", index=False)