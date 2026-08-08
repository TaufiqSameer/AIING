import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


train_df = pd.read_csv("train.csv")


# --------------------------------------------------
# Sentence splitter
# --------------------------------------------------

# def split_sentences(text):
#     s = re.split(r'(?<=[.!?])\s+', str(text).strip())
#     return [x.strip() for x in s if x.strip()]


# def only_second(text):
#     s = split_sentences(text)
#     return s[1] if len(s) >= 2 else ""


# def without_second(text):
#     s = split_sentences(text)
#     if len(s) < 2:
#         return " ".join(s)
#     return " ".join(s[:1] + s[2:])


# # --------------------------------------------------
# # Build three datasets
# # --------------------------------------------------

# datasets = {
#     "Full Review": train_df["Reviews"],
#     "Only Second Sentence": train_df["Reviews"].apply(only_second),
#     "Without Second Sentence": train_df["Reviews"].apply(without_second),
# }


# # --------------------------------------------------
# # Common split
# # --------------------------------------------------

# train_idx, val_idx = train_test_split(
#     train_df.index,
#     test_size=0.2,
#     random_state=42,
#     stratify=train_df["Course"],
# )


# # --------------------------------------------------
# # Train & Evaluate
# # --------------------------------------------------

# for name, text in datasets.items():

#     X_train = text.loc[train_idx]
#     X_val = text.loc[val_idx]

#     y_train = train_df.loc[train_idx, "Course"]
#     y_val = train_df.loc[val_idx, "Course"]

#     clf = Pipeline([
#         (
#             "tfidf",
#             TfidfVectorizer(
#                 stop_words="english",
#                 ngram_range=(1,3),
#                 binary=True,
#                 sublinear_tf=True,
#                 min_df=2,
#                 max_df=0.9,
#             ),
#         ),
#         (
#             "lr",
#             LogisticRegression(max_iter=1000),
#         ),
#     ])

#     clf.fit(X_train, y_train)

#     pred = clf.predict(X_val)

#     print("=" * 60)
#     print(name)
#     print("Accuracy:", accuracy_score(y_val, pred))

course = "Flask API Development"

sample = train_df[
    train_df.Course == course
].sample(20, random_state=42)

for i, review in enumerate(sample["Reviews"]):
    print("="*80)
    print(i)
    print(review)