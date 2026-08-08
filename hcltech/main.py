import numpy as np;
import pandas as pd;
import seaborn as sns;
import matplotlib.pyplot as plt;
import re

def split_sentences(text):
    return frozenset(
        s.strip()
        for s in re.split(r'(?<=[.!?])\s+', text)
        if s.strip()
    )


train_df = pd.read_csv("train.csv");
test_df = pd.read_csv("test.csv");
train_sentence_sets = [split_sentences(review)for review in train_df["Reviews"]]

# print(train_df.head());
# print(test_df.head());

# print(train_df.shape);
# print(test_df.shape);


# print(train_df.info);
# print(test_df.info);

# print(train_df.isnull().sum()) 
# print(test_df.isnull().sum()) 

# print("Duplicate rows:", train_df.duplicated().sum())

# print("Duplicate reviews:",
#       train_df["Reviews"].duplicated().sum())

# print(train_df["Course"].nunique())

# print(train_df["Course"].value_counts())

course = train_df["Course"].value_counts()

plt.figure(figsize=(14,6))

course.plot(kind="bar")

plt.title("reviews/course")
plt.ylabel("cnt")
plt.xticks()

plt.tight_layout()
# plt.show()

train_df["char_length"] = train_df["Reviews"].str.len()

train_df["word_length"] = train_df["Reviews"].str.split().str.len()

# print(train_df[["char_length","word_length"]].describe());

from collections import Counter

words = " ".join(train_df["Reviews"]).lower().split()
counter = Counter(words)


# print(counter.most_common(30))

# print(train_df["Course"].value_counts().describe())

course = train_df["Course"].iloc[0]

# for c in train_df["Course"]:
#     print("The course we are seeing is " , c )
    
#     sample = train_df[train_df["Course"] == c]["Reviews"].head(1)
    
#     for review in sample:
#         print(review)
#         print("-"*80)
        
# c = input("enter the course name");

# sample = train_df[train_df["Course"] == c]["Reviews"].head(1)

# for s in sample:
#         print(s)
#         print("-"*80)
        
# dup = (
#     train_df.groupby("Reviews")["Course"]
#          .nunique()
#          .sort_values(ascending=False)
# )

# print(dup.head(20))

# from sklearn.feature_extraction.text import CountVectorizer

# vectorizer = CountVectorizer(stop_words='english')

# X = vectorizer.fit_transform(train_df["Reviews"])

# word_freq = np.asarray(X.sum(axis=0)).ravel()

# words = vectorizer.get_feature_names_out()

# # freq = pd.DataFrame({
# #     "word": words,
# #     "count": word_freq
# # }).sort_values("count", ascending=False)

# # print(freq.head(50))

# # from sklearn.feature_extraction.text import CountVectorizer

# # course = "AWS Cloud Practitioner"

# # text = " ".join(train_df[train_df["Course"] == course]["Reviews"])

# # vectorizer = CountVectorizer(stop_words="english")

# # X = vectorizer.fit_transform([text])

# # words = vectorizer.get_feature_names_out()

# # counts = X.toarray()[0]

# # freq = pd.DataFrame({
# #     "word_aws": words,
# #     "count_aws": counts
# # }).sort_values("count_aws", ascending=False)

# # print(freq.head(30))
# # imp
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel
# from collections import Counter

# vectorizer = TfidfVectorizer(
#     stop_words="english",
#     ngram_range=(1, 3),
#     min_df=2,
#     max_df=0.8,
#     sublinear_tf=True,
# )
# char_vectorizer = TfidfVectorizer(
#     analyzer="char_wb",
#     ngram_range=(3,5),
#     min_df=3,
#     sublinear_tf=True,
# )

# X_train_word = vectorizer.fit_transform(train_df["Reviews"])
# X_test_word = vectorizer.transform(test_df["Reviews"])

# X_train_char = char_vectorizer.fit_transform(train_df["Reviews"])
# X_test_char = char_vectorizer.transform(test_df["Reviews"])

# print(X_train_word.shape)
# print(X_train_char.shape)
# X_train = vectorizer.fit_transform(train_df["Reviews"])
# print("Shape:", X_train.shape)
# X_test = vectorizer.transform(test_df["Reviews"])
# print(len(vectorizer.vocabulary_))

# # from tqdm import tqdm

# # predictions = []

# # i = 0

# # similarity = cosine_similarity(
# #     X_test[i],
# #     X_train
# # ).flatten()

# # top10 = np.argpartition(similarity, -10)[-10:]
# # top10 = top10[np.argsort(similarity[top10])[::-1]]

# # print("Test Review:\n")
# # print(test_df.iloc[i]["Reviews"])

# # print("\nTop 10 Matches:\n")

# # print(train_df.iloc[top10][["Course", "Reviews"]])

# # for l in top10:
# #     print("=" * 80)
# #     print(train_df.iloc[l]["Course"])
# #     print()
# #     print(train_df.iloc[l]["Reviews"])
# #     print()
# train_courses = train_df["Course"].values
# train_indices = train_df["Index"].values

# predictions = []

# # i = 0

# # smi = linear_kernel(X_test[i], X_train).ravel()

# # top100 = np.argpartition(smi, -100)[-100:]
# # top100 = top100[np.argsort(smi[top100])[::-1]]

# # courses = train_df.iloc[top100]["Course"]

# # print(courses.value_counts())
# # import random

# # for i in random.sample(range(X_test.shape[0]), 20):

# #     smi = linear_kernel(X_test[i], X_train).ravel()

# #     top100 = np.argpartition(smi, -100)[-100:]
# #     top100 = top100[np.argsort(smi[top100])[::-1]]

# #     counts = train_df.iloc[top100]["Course"].value_counts()

# #     print(f"Test review {i}")
# #     print(counts.head())
# #     print("-" * 60)
# from tqdm import tqdm
# #imp
# # for i in tqdm(range(X_test.shape[0])):

# #     smi = linear_kernel(X_test[i], X_train).ravel()

# #     top10 = np.argpartition(smi, -10)[-10:]

# #     top10 = top10[np.argsort(smi[top10])]
# #     top10 = top10[::-1]

# #     recom = train_df.iloc[top10]["Index"].tolist()
# #     predictions.append(str(recom))
#     # print(type(recom))
#     # predictions.append(str(recom))
    
# # batch_size = 256

# # predictions = []

# # for start in tqdm(range(0, X_test_word.shape[0], batch_size)):

# #     end = min(start + batch_size, X_test_word.shape[0])

# #     word_sim = linear_kernel(
# #         X_test_word[start:end],
# #         X_train_word
# #     )
# #     char_sim = linear_kernel(
# #         X_test_char[start:end],
# #         X_train_char
# #     )
# #     sim = 0.7 * word_sim + 0.3 * char_sim
# #     for row in sim:

# #         top10 = np.argpartition(row, -10)[-10:]
# #         top10 = top10[np.argsort(row[top10])[::-1]]

# #         recom = train_indices[top10].tolist()
# #         predictions.append(str(recom))

# # for i in tqdm(range(X_test.shape[0])):

# #     smi = linear_kernel(X_test[i], X_train).ravel()

# #     
# #     top100 = np.argpartition(smi, -100)[-100:]
# #     top100 = top100[np.argsort(smi[top100])[::-1]]

# #     
# #     courses = train_courses[top100]
# #     majority_course = Counter(courses).most_common(1)[0][0]

# #    
# #     filtered = [idx for idx in top100 if train_courses[idx] == majority_course]

# #    
# #     if len(filtered) < 10:
# #         used = set(filtered)
# #         for idx in top100:
# #             if idx not in used:
# #                 filtered.append(idx)
# #             if len(filtered) == 10:
# #                 break

# #     recom = train_indices[filtered[:10]].tolist()
# #     predictions.append(str(recom))
    

# # print(train_df["Course"].nunique()) 
# # # import torch
# # print(train_df["Course"].value_counts().head(10))
# # print("grahpic card available : " ,torch.cuda.zis_available());
# # print("which one " , torch.cuda.get_device_capability());
# # print("which one " , torch.cuda.get_device_properties());

# # from sentence_transformers import SentenceTransformer

# # model = SentenceTransformer("all-MiniLM-L6-v2",device="cuda"
# # )

# # print(model.device)
# # train_reviews = train_df["Reviews"].fillna("").tolist()
# # test_reviews = test_df["Reviews"].fillna("").tolist()

# # training = model.encode(train_reviews,batch_size=128,show_progress_bar=True,convert_to_numpy=True,normalize_embeddings=True
# # )
# # print(training.shape)

# # testing = model.encode(test_reviews,batch_size=128,show_progress_bar=True,convert_to_numpy=True,normalize_embeddings=True
# # )
# # print(testing.shape)
# # import faiss
# # index = faiss.IndexFlatIP(384)

# # index.add(training.astype(np.float32))
# # print(index.ntotal)

# # k = 10
# # distances, indices = index.search(testing.astype(np.float32), k)

# # predictions = []

# # for i in indices:
# #     retrieved = train_df.iloc[i]["Index"].tolist()
# #     predictions.append(str(retrieved))

# # submission = pd.DataFrame({
# #     "Index": test_df["Index"],
# #     "Index_list": predictions
# # })

# # submission.to_csv("submission.csv", index=False)

# # print(submission.shape)

# # from rank_bm25 import BM25Okapi

# # train_tokens = [review.lower().split()for review in train_df["Reviews"]]

# # test_tokens = [review.lower().split()for review in test_df["Reviews"]]

# # bm25 = BM25Okapi(train_tokens)

# # from tqdm import tqdm

# # predictions = []

# # for query in tqdm(test_tokens[:100]):
# #     scores = bm25.get_scores(query)

# #     top10 = np.argpartition(scores, -10)[-10:]
# #     top10 = top10[np.argsort(scores[top10])[::-1]]

# #     recom = train_df.iloc[top10]["Index"].tolist()
# #     predictions.append(str(recom))

# # predictions = []

# # for query in tqdm(test_tokens):

# #     scores = bm25.get_scores(query)

# #     top10 = np.argpartition(scores, -10)[-10:]
# #     top10 = top10[np.argsort(scores[top10])[::-1]]

# #     recom = train_df.iloc[top10]["Index"].tolist()
# #     predictions.append(str(recom))

# from sklearn.neighbors import NearestNeighbors

# # nn = NearestNeighbors(
# #     n_neighbors=10,
# #     metric="euclidean",
# #     algorithm="brute",
# #     n_jobs=-1
# # )

# # nn.fit(X_train)

# # distances, indices = nn.kneighbors(X_test)
# nn = NearestNeighbors(
#     n_neighbors=20,
#     metric="cosine",
#     algorithm="brute",
#     n_jobs=-1
# )

# nn.fit(X_train)

# distances, indices = nn.kneighbors(X_test)

# ss = pd.read_csv("sample_submission.csv")
# submission = ss.copy()


# predictions = []
# distances, indices = nn.kneighbors(X_test)

# # predictions = []

# # for i, idx in enumerate(indices):
# #     char_scores = linear_kernel(
# #         X_test_char[i],
# #         X_train_char[idx]
# #     ).ravel()
# #     word_scores = 1 - distances[i] 
# #     final_scores = 0.7 * word_scores + 0.3 * char_scores

# #     best10 = idx[np.argsort(final_scores)[-10:][::-1]]

# #     predictions.append(str(train_indices[best10].tolist()))
# # import random

# # for i in random.sample(range(len(test_df)), 50):
# #     idx = indices[i]    
# #     courses = train_df.iloc[idx]["Course"].nunique()
# #     print(courses)
    
# for idx in indices:
#     predictions.append(str(train_indices[idx].tolist()))

# predictions = []
# # i = 0

# # print("TEST REVIEW")
# # print(test_df.iloc[i]["Reviews"])

# # print("=" * 80)

# # for rank, idx in enumerate(indices[i]):
# #     print(f"Rank {rank+1}")
# #     print("Similarity:", 1 - distances[i][rank])
# #     print(train_df.iloc[idx]["Reviews"])
# #     print("-" * 80)

# submission = pd.DataFrame({
#     "Index": test_df["Index"],
#     "Index_list": predictions
# })

# submission.to_csv("knnreranking.csv", index=False)
# print(submission.shape); 

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.neighbors import NearestNeighbors

# vectorizer = TfidfVectorizer(
#     stop_words="english",
#     ngram_range=(1,3),
#     min_df=2,
#     max_df=0.9,
#     sublinear_tf=True, 
#     binary=True
# )


# X_train = vectorizer.fit_transform(train_df["Reviews"])
# X_test = vectorizer.transform(test_df["Reviews"])

# nn = NearestNeighbors(
#     metric="cosine",
#     algorithm="brute",
#     n_neighbors=50,
#     n_jobs=-1,
    
# )
for i, idxs in enumerate(indices):
#     # True course from the second sentence
#     second = get_second_sentence(test_df.iloc[i]["Reviews"])
#     true_course = course_lookup[second]

#     # Course of the nearest neighbour
#     predicted_course = train_courses[idxs[0]]

#     if predicted_course == true_course:
#         correct_course += 1

# print(correct_course, "/", len(test_df))
# print(correct_course / len(test_df))

# nn.fit(X_train)

# distances, indices = nn.kneighbors(X_test)

# train_indices = train_df["Index"].values

# predictions = [
#     str(train_indices[idx].tolist())
#     for idx in indices
# ]

# submission = pd.DataFrame({
#     "Index": test_df["Index"],
#     "Index_list": predictions
# })

import re

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

# missing = 0

# for review in test_df["Reviews"]:
#     second = get_second_sentence(review)

#     if second not in course_lookup:
#         missing += 1

# print("Missing:", missing)

# course_to_indices = {}

# for idx, course in enumerate(train_df["Course"]):
#     course_to_indices.setdefault(course, []).append(idx)
# submission.to_csv("submissio_lol.csv", index=False)

# train_courses = train_df["Course"].values

# correct_course = 0

# 
def weighted_review(text):
    s = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(s) < 7:
        return text
    parts = []
    parts.append(s[0])
    parts.extend([s[1]] * 4)
    parts.extend([s[2]] * 3)
    parts.extend([s[3]] * 2)
    parts.extend(s[4:])

    return " ".join(parts)

train_text = train_df["Reviews"].apply(weighted_review)
test_text = test_df["Reviews"].apply(weighted_review)

train_sentences = pd.Series(train_text).map(split)
test_sentences = pd.Series(test_text).map(split)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

vectorizer = CountVectorizer(
    stop_words="english",
    ngram_range=(1,3),
    min_df=2,
    max_df=0.9,
    binary=True,
)

X_train = vectorizer.fit_transform(train_df["Reviews"])
X_test = vectorizer.transform(test_df["Reviews"])

nn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=10
)

nn.fit(X_train)

distances, indices = nn.kneighbors(X_test)

train_indices = train_df["Index"].values

predictions = [
    str(train_indices[idx].tolist())
    for idx in indices
]


submission = pd.DataFrame({
    "Index": test_df["Index"],
    "Index_list": predictions
})

submission.to_csv("submission_imtired.csv", index=False)