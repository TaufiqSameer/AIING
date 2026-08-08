import pandas as pd
import re
import matplotlib.pyplot as plt;
import seaborn as sns;

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
import re


def tag_sentence(sentence, tag):
    words = re.findall(r"\b\w+\b", sentence.lower())
    return " ".join(f"{tag}_{w}" for w in words)

def structured_review(review, course):
    sentences = re.split(r'(?<=[.!?])\s+', review)
    if sentences:
        sentences[0] = re.sub(
            re.escape(course),
            "this course",
            sentences[0],
            flags=re.IGNORECASE,
        )
    tagged = []
    for i, sentence in enumerate(sentences):
        tag = f"s{i+1}"
        tagged_sentence = tag_sentence(sentence, tag)
        tagged.append(tagged_sentence)
        if i == 1:
            tagged.append(tagged_sentence)

    return " ".join(tagged)

train_df["Reviews"] = [
    structured_review(review, course)
    for review, course in zip(train_df["Reviews"], train_df["Course"])
]


test_df["Reviews"] = [
    structured_review(review, "")
    for review in test_df["Reviews"]
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
    n_neighbors=10,
)

nn.fit(X_train)

distances, indices = nn.kneighbors(X_test)

submission = pd.DataFrame({
    "Index": test_df["Index"],
    "Index_list": [
        str(list(train_df.iloc[idx]["Index"]))
        for idx in indices
    ]
})

submission.to_csv("submission_structured.csv", index=False)

