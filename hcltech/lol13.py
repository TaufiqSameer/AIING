import pandas as pd
import matplotlib.pyplot as plt;
import seaborn as sns;
import re
import numpy as np;
from tqdm import tqdm;

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# old = pd.read_csv("eighty85.csv")
# new = pd.read_csv("eighty87.csv")

# print((old["Index_list"] != new["Index_list"]).sum())

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
from sklearn.metrics.pairwise import cosine_similarity

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

NUM_SENTENCES = 6


def split_sentences(text):
    s = re.split(r'(?<=[.!?])\s+', text.strip())

    while len(s) < NUM_SENTENCES:
        s.append("")

    return s[:NUM_SENTENCES]


train_sentences = list(map(split_sentences, train_text))
test_sentences = list(map(split_sentences, test_text))

word_vectorizers = []
char_vectorizers = []

train_word_vectors = []
train_char_vectors = []

for pos in range(NUM_SENTENCES):

    train_docs = [x[pos] for x in train_sentences]

    word_vec = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        binary=False,
        sublinear_tf=True,
        ngram_range=(1,3),
        min_df=2,
        max_df=0.9,
    )

    X_word = word_vec.fit_transform(train_docs)

    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3,5),
        sublinear_tf=True,
    )

    X_char = char_vec.fit_transform(train_docs)

    word_vectorizers.append(word_vec)
    char_vectorizers.append(char_vec)

    train_word_vectors.append(X_word)
    train_char_vectors.append(X_char)

weights = np.array([
    3,
    3,
    2,
    2,
    1,
    1
])

final_indices = []

for q in tqdm(range(len(test_df))):

    candidates = indices[q]

    score = np.zeros(len(candidates))

    query_sentences = split_sentences(test_text[q])

    for pos in range(NUM_SENTENCES):


        query_word = word_vectorizers[pos].transform(
            [query_sentences[pos]]
        )

        word_sim = cosine_similarity(
            query_word,
            train_word_vectors[pos][candidates]
        ).ravel()



        query_char = char_vectorizers[pos].transform(
            [query_sentences[pos]]
        )

        char_sim = cosine_similarity(
            query_char,
            train_char_vectors[pos][candidates]
        ).ravel()
        sentence_score = (
            0.7 * word_sim +
            0.3 * char_sim
        )

        score += weights[pos] * sentence_score

    order = np.argsort(-score)

    final_indices.append(
        candidates[order[:10]]
    )

submission = pd.DataFrame({
    "Index": test_df["Index"],
    "Index_list": [
        str([int(x) for x in row])
        for row in final_indices
    ]
})

submission.to_csv("sentence_rerank.csv", index=False)

print(submission.head())
# from collections import Counter

# course_distribution = []

# for row in indices:
#     courses = train_df.iloc[row]["Course"]

#     course_distribution.append(
#         Counter(courses).most_common(5)
#     )

# print("\nFirst 20 queries:\n")

# for i in range(20):
#     print(f"Query {i}:")
#     print(course_distribution[i])
#     print()

# print("=" * 60)

# all_same = sum(
#     1
#     for x in course_distribution
#     if x[0][1] == 100
# )

# print("Queries whose Top100 are ALL from one course:", all_same)
# print("Percentage:", all_same / len(course_distribution) * 100)

# print("=" * 60)

# not_all_same = [
#     x
#     for x in course_distribution
#     if x[0][1] != 100
# ]

# print("Examples where Top100 contain multiple courses:\n")

# for x in not_all_same[:20]:
#     print(x)

# def split_sentences(text):
#     s = re.split(r'(?<=[.!?])\s+', text)

#     while len(s) < 6:
#         s.append("")

#     return s[:6]
# from collections import Counter

# for q in range(10):

#     row = indices[q]

#     print("=" * 80)
#     print("QUERY", q)
#     print("=" * 80)

#     for rank, idx in enumerate(row[:10], 1):
#         print(f"\nRank {rank}")
#         print(train_df.iloc[idx]["Reviews"])

# train_sentences = list(map(split_sentences, train_text))
# test_sentences = list(map(split_sentences, test_text))

# sentence_vectorizers = []
# train_sentence_vectors = []

# for pos in range(6):

#     docs = [x[pos] for x in train_sentences]

#     tfidf = TfidfVectorizer(
#         stop_words="english",
#         ngram_range=(1,3),
#         binary=False,
#         sublinear_tf=True,
#         min_df=2,
#         max_df=0.9,
#     )

#     X = tfidf.fit_transform(docs)

#     sentence_vectorizers.append(tfidf)
#     train_sentence_vectors.append(X)
    
# final_indices = []

# weights = np.array([
#     3,
#     2,
#     1,
#     1,
#     1,
#     1,
# ])

# for q in tqdm(range(len(test_df))):

#     candidates = indices[q]

#     score = np.zeros(len(candidates))

#     sents = split_sentences(test_text[q])

#     for pos in range(6):

#         query = sentence_vectorizers[pos].transform(
#             [sents[pos]]
#         )

#         sims = (
#             train_sentence_vectors[pos][candidates]
#             @ query.T
#         ).toarray().ravel()

#         score += weights[pos] * sims

#     order = np.argsort(-score)

#     final_indices.append(
#         candidates[order[:10]]
#     )
# submission = pd.DataFrame({
#     "Index": test_df["Index"],
#     "Index_list": [
#         str([int(x) for x in row])
#         for row in final_indices
#     ]
# })

# submission.to_csv("eighty88.csv", index=False)

# print(submission.head())