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
def anonymize_first_sentence(review, course):
    sentences = re.split(r'(?<=[.!?])\s+', review)
    if sentences:
        pattern = re.escape(course)
        sentences[0] = re.sub(pattern,"this course",
            sentences[0],
            flags=re.IGNORECASE,
        )

    return " ".join(sentences)

train_df["Body"] = (
    train_df["S2"] + " " +
    train_df["S3"] + " " +
    train_df["S4"] + " " +
    train_df["S5"] + " " +
    train_df["S6"]
)

test_df["Body"] = (
    test_df["S2"] + " " +
    test_df["S3"] + " " +
    test_df["S4"] + " " +
    test_df["S5"] + " " +
    test_df["S6"]
)
train_df["AnonReview"] = train_df.apply(
    lambda row: anonymize_first_sentence(
        row["Reviews"],
        row["Course"]
    ),
    axis=1
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3,5),
    binary=True,
    sublinear_tf=True,
    lowercase=True,
)

X_train = vectorizer.fit_transform(train_df["AnonReview"])
X_test = vectorizer.transform(test_df["Reviews"])

nn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=10
)

nn.fit(X_train)
distances, indices = nn.kneighbors(X_test)

submission = pd.DataFrame({
    "Index": test_df["Index"],
    "Index_list": [
        str(train_df.iloc[idx]["Index"].tolist())
        for idx in indices
    ]
})

submission.to_csv("submission_char.csv", index=False)