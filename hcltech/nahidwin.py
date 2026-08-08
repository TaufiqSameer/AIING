from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt;
import seaborn as sns;
import re

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def split_sentences(text):
    s = re.split(r'(?<=[.!?])\s+', text)

    while len(s) < 6:
        s.append("")

    return s[:6]


train_sentences = list(train_df["Reviews"].apply(split_sentences))
test_sentences = list(test_df["Reviews"].apply(split_sentences))

vectorizers = []
train_vectors = []
test_vectors = []

for pos in tqdm(range(6)):

    train_docs = [x[pos] for x in train_sentences]
    test_docs = [x[pos] for x in test_sentences]

    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1,3),
        binary=True,
        sublinear_tf=True,
        min_df=2,
        max_df=0.9,
    )

    X_train = tfidf.fit_transform(train_docs)
    X_test = tfidf.transform(test_docs)

    vectorizers.append(tfidf)
    train_vectors.append(X_train)
    test_vectors.append(X_test)
import numpy as np;
weights = np.array([
    1.0,   
    1.5,   
    2.0,   
    2.0,   
    2.5,   
    2.5    
])

scores = np.zeros((len(test_df), len(train_df)), dtype=np.float32)
for i in tqdm(range(6)):

    sim = cosine_similarity(
        test_vectors[i],
        train_vectors[i],
        dense_output=True,
    )

    scores += weights[i] * sim
    
top10 = np.argsort(-scores, axis=1)[:, :10]

submission = pd.DataFrame({
    "Index": test_df["Index"],
    "Index_list": [
        str([int(x) for x in row])
        for row in top10
    ]
})

submission.to_csv("submission_sentencewise.csv", index=False)