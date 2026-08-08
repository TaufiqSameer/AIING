import pandas as pd
import matplotlib.pyplot as plt;
import seaborn as sns;
import re
import numpy as np;
from tqdm import tqdm;

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
def split(review):
    s = re.split(r'(?<=[.!?])\s+', review.strip())
    while len(s) < 6:
        s.append("")
    return s[:6]
train_sentences = train_df["Reviews"].map(split)
template_maps = []
train_template_ids = []

for pos in range(2, 6):     

    mp = {}
    ids = []

    for review in train_sentences:

        sent = review[pos]

        if sent not in mp:
            mp[sent] = len(mp)

        ids.append(mp[sent])

    template_maps.append(mp)
    train_template_ids.append(np.array(ids))
    
from collections import Counter

template_weights = []

for ids in train_template_ids:

    freq = Counter(ids)

    total = len(ids)

    weight = {}

    for tid, cnt in freq.items():
        weight[tid] = np.log1p(total / cnt)

    template_weights.append(weight)


test_sentences = test_df["Reviews"].map(split)

test_template_ids = []

for pos in range(2,6):

    mp = template_maps[pos-2]

    ids = []

    for review in test_sentences:

        ids.append(
            mp.get(review[pos], -1)
        )

    test_template_ids.append(np.array(ids))

def anonymize_first_sentence(review, course):
    sentences = re.split(r'(?<=[.!?])\s+', review)
    if sentences:
        pattern = re.escape(course)
        sentences[0] = re.sub(pattern,"this course",
            sentences[0],
            flags=re.IGNORECASE,
        )

    return " ".join(sentences)

def preprocess_train(review, course):
    review = anonymize_first_sentence(review, course)

    sentences = re.split(r'(?<=[.!?])\s+', review)

    if len(sentences):
        review = " ".join([sentences[0], sentences[0]] + sentences[1:])

    return review
# X_train_char = char_vectorizer.fit_transform(train_df["Reviews"])
# X_test_char = char_vectorizer.transform(test_df["Reviews"])

# print("Comp2")
# from scipy.sparse import hstack

# X_train = hstack([X_train_word, X_train_char])
# X_test = hstack([X_test_word, X_test_char])

# import time

# nn = NearestNeighbors(
#     metric="cosine",
#     algorithm="brute",
#     n_neighbors=10,
# )

# start = time.time()
# print("Comp3")
# nn.fit(X_train)
# distances, indices = nn.kneighbors(X_test, n_neighbors=100)

# unique_course_counts = []

# for row in indices:
#     courses = train_df.iloc[row]["Course"]
#     unique_course_counts.append(courses.nunique())

# import pandas as pd

# print(pd.Series(unique_course_counts).describe())

# print(pd.Series(unique_course_counts).value_counts().sort_index())

# import matplotlib.pyplot as plt
# import numpy as np

# top1 = distances[:,0]
# top2 = distances[:,1]
# top10 = distances[:,9]

# plt.figure(figsize=(8,5))
# plt.hist(top1, bins=100, alpha=0.6, label="Top1")
# plt.hist(top2, bins=100, alpha=0.6, label="Top2")
# plt.hist(top10, bins=100, alpha=0.6, label="Top10")
# plt.legend()
# plt.xlabel("Cosine Distance")
# plt.ylabel("Count")
# plt.show()

# def preprocess_test(review):
#     sentences = re.split(r'(?<=[.!?])\s+', review)

#     if len(sentences):
#         review = " ".join([sentences[0], sentences[0]] + sentences[1:])

#     return review

# train_text = [
#     preprocess_train(r, c)
#     for r, c in tqdm(zip(train_df["Reviews"], train_df["Course"]),
#                      total=len(train_df))
# ]


def preprocess_test(review):
    sentences = re.split(r'(?<=[.!?])\s+', review)

    if len(sentences):
        review = " ".join([sentences[0], sentences[0]] + sentences[1:])

    return review

train_text = [
    preprocess_train(r, c)
    for r, c in tqdm(zip(train_df["Reviews"], train_df["Course"]),
                     total=len(train_df))
]



# def get_second_sentence(text):
#     sents = re.split(r'(?<=[.!?])\s+', text.strip())
#     return sents[1] if len(sents) > 1 else ""

# course_lookup = {}

# for _, row in train_df.iterrows():
#     second = get_second_sentence(row["Reviews"])
#     course = row["Course"]

#     if second in course_lookup and course_lookup[second] != course:
#         print("Conflict:", second)

#     course_lookup[second] = course
    
# missing = 0

# for review in test_df["Reviews"]:
#     second = get_second_sentence(review)

#     if second not in course_lookup:
#         missing += 1

# print("Missing:", missing)
test_text = [
    preprocess_test(r)
    for r in tqdm(test_df["Reviews"])
]


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,3),
    binary=False,
    sublinear_tf=True,
    min_df=2,
    max_df=0.9,
)

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3,5),
    sublinear_tf=True,
)

X_train_word = vectorizer.fit_transform(train_text)
X_test_word = vectorizer.transform(test_text)

X_train_char = char_vectorizer.fit_transform(train_text)
X_test_char = char_vectorizer.transform(test_text)

from scipy.sparse import hstack
X_train = hstack([
    X_train_word,
    X_train_char * 0.55
])

X_test = hstack([
    X_test_word,
    X_test_char 
])


nn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=10,
)

nn.fit(X_train)

distances, indices = nn.kneighbors(
    X_test,
    n_neighbors=100
)

final_indices = []

BOOST = 0.02

for q in tqdm(range(len(test_df))):

    candidates = indices[q]

    score = -distances[q].copy()

    for k, idx in enumerate(candidates):

        bonus = 0.0

        for pos in range(4):

            test_tid = test_template_ids[pos][q]

            if test_tid == -1:
                continue

            if test_tid == train_template_ids[pos][idx]:

                bonus += template_weights[pos][test_tid]

        score[k] += bonus * BOOST

    order = np.argsort(-score)

    final_indices.append(candidates[order[:10]])
submission = pd.DataFrame({
    "Index": test_df["Index"],
    "Index_list": [
        str([int(x) for x in row])
        for row in final_indices
    ]
})
submission.to_csv("vootukuri.csv", index=False)

print(submission.head())