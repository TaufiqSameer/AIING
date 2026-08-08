import pandas as pd
import re
import matplotlib.pyplot as plt;
import seaborn as sns;
import numpy as np;

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
        sentences[0] = re.sub(
            pattern,
            "",
            sentences[0],
            flags=re.IGNORECASE,
        )

        sentences[0] = re.sub(r"\s+", " ", sentences[0]).strip()

        sentences[0] = re.sub(r"\s+([.,!?])", r"\1", sentences[0])

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
char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3,5),
    sublinear_tf=True,
)
X_train = tfidf.fit_transform(train_df["Reviews"])
X_test = tfidf.transform(test_df["Reviews"])
X_train_char = char_vectorizer.fit_transform(train_df["Reviews"])
X_test_char = char_vectorizer.transform(test_df["Reviews"])

nn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=100
)

nn.fit(X_train)
distances, indices = nn.kneighbors(X_test)
from sklearn.metrics.pairwise import cosine_similarity

predictions = []

from tqdm import tqdm

for i in tqdm(range(len(test_df))):

    candidates = indices[i]

    sims = cosine_similarity(
        X_test_char[i],
        X_train_char[candidates]
    ).ravel()

    order = np.argsort(-sims)

    best10 = candidates[order[:10]]

    predictions.append(
        [int(train_df.iloc[j]["Index"]) for j in best10]
    )
submission = pd.DataFrame({
    "Index": test_df["Index"],
    "Index_list": [str(x) for x in predictions]
})
submission.to_csv("submissionll2.csv", index=False)