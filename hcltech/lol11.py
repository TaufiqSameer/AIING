import pandas as pd
import matplotlib.pyplot as plt;
import seaborn as sns;
import re
import numpy as np;
from tqdm import tqdm;
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

def remaining_sentences(review):
    sentences = re.split(r'(?<=[.!?])\s+', review)
    return " ".join(sentences[1:]) if len(sentences) > 1 else ""

def anonymize_first_sentence(review, course):
    sentences = re.split(r'(?<=[.!?])\s+', review)
    if sentences:
        pattern = re.escape(course)
        sentences[0] = re.sub(pattern,"this course",
            sentences[0],
            flags=re.IGNORECASE,
        )

    return " ".join(sentences)

def preprocess_test(review):
    sentences = re.split(r'(?<=[.!?])\s+', review)

    if len(sentences):
        review = " ".join([sentences[0], sentences[0]] + sentences[1:])

    return review


def preprocess_train(review, course):
    review = anonymize_first_sentence(review, course)

    sentences = re.split(r'(?<=[.!?])\s+', review)

    if len(sentences):
        review = " ".join([sentences[0], sentences[0]] + sentences[1:])

    return review

def first_sentence(review):
    sentences = re.split(r'(?<=[.!?])\s+', review)
    return sentences[0] if sentences else review

train_full = [preprocess_train(r, c)for r, c in zip(train_df["Reviews"], train_df["Course"])]

test_full = [preprocess_test(r)for r in test_df["Reviews"]]
train_rest = [remaining_sentences(x) for x in train_full]
test_rest = [remaining_sentences(x) for x in test_full]


train_s1 = [first_sentence(x) for x in train_full]
test_s1 = [first_sentence(x) for x in test_full]

word_full = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,3),
    binary=False,
    sublinear_tf=True,
    min_df=2,
    max_df=0.9,
)

word_s1 = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,3),
    binary=False,
    sublinear_tf=True,
    min_df=2,
    max_df=0.9,
)

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(4,6),
    sublinear_tf=True,
)

X_train_full = word_full.fit_transform(train_full)
X_test_full = word_full.transform(test_full)

X_train_s1 = word_s1.fit_transform(train_s1)
X_test_s1 = word_s1.transform(test_s1)

X_train_char = char_vectorizer.fit_transform(train_full)
X_test_char = char_vectorizer.transform(test_full)

X_train = hstack([
    X_train_s1,
    X_train_full,
    X_train_char * 0.55,
])

X_test = hstack([
    X_test_s1,
    X_test_full,
    X_test_char * 0.55,
])


nn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=10,
)

nn.fit(X_train)

_, indices = nn.kneighbors(X_test)

submission = pd.DataFrame({
    "Index": test_df["Index"],
    "Index_list": [
        str([int(x) for x in row])
        for row in indices
    ]
})

submission.to_csv("eighty86.csv", index=False)

print(submission.head())