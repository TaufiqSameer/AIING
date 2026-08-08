from collections import defaultdict,Counter
import numpy as np;
import pandas as pd;
import seaborn as sns;
import matplotlib.pyplot as plt;
from tqdm import tqdm;

train_df = pd.read_csv("train.csv");
test_df = pd.read_csv("test.csv");

# import re

# def split_review(text):
#     return re.split(r'(?<=[.!?])\s+', text.strip())

# lengths = Counter()

# for review in train_df["Reviews"]:
#     lengths[len(split_review(review))] += 1

# print(lengths)
# slot_values = defaultdict(set)

# for review in train_df["Reviews"]:
#     s = split_review(review)

#     for i, sentence in enumerate(s):
#         slot_values[i].add(sentence)

# for slot in sorted(slot_values):
#     print(slot, len(slot_values[slot]))

# slot_maps = {}

# for slot in slot_values:
#     slot_maps[slot] = {
#         sentence:i
#         for i,sentence in enumerate(sorted(slot_values[slot]))
#     }
# encoded = []

# for review in train_df["Reviews"]:
#     s = split_review(review)

#     encoded.append(
#         tuple(slot_maps[i][x] for i,x in enumerate(s))
#     )
# cnt = Counter(encoded)

# # print(cnt.most_common(20))

# def hamming(a,b):
#     return sum(x!=y for x,y in zip(a,b))

# import re

# for review in train_df["Reviews"].head(20):
#     print("="*80)

#     s = re.split(r'(?<=[.!?])\s+', review.strip())

#     for i,x in enumerate(s):
#         print(i, x)

# from collections import defaultdict
# import re


# banks = defaultdict(dict)
# def split_review(text):
#     return re.split(r'(?<=[.!?])\s+', text.strip())

# def fit_encoder(reviews):
#     for review in reviews:
#         s = split_review(review)

#         for pos, sent in enumerate(s):
#             if sent not in banks[pos]:
#                 banks[pos][sent] = len(banks[pos])

# fit_encoder(train_df.Reviews)

# def encode(review):
#     s = split_review(review)

#     ids = []

#     for i, sent in enumerate(s):
#         if i >= len(template_maps):
#             break
#         ids.append(template_maps[i].get(sent, -1))

#     return tuple(ids)

# train_encoded = train_df.Reviews.apply(encode)
# test_encoded = test_df.Reviews.apply(encode)

# from collections import Counter
# import math

# slot_freq = {}

# N = len(train_encoded)

# for slot in range(2,7):

#     freq = Counter()

#     for x in train_encoded:

#         if slot < len(x):
#             freq[x[slot]] += 1

#     slot_freq[slot] = freq
    
# def rarity(slot,value):

#     freq = slot_freq[slot][value]

#     return math.log(N/freq)

# def features(query,candidate,cosine):

#     f = {}

#     f["cosine"] = cosine

#     f["length_diff"] = abs(len(query)-len(candidate))

#     matches = 0
#     rarity_score = 0

#     for slot in range(2,min(len(query),len(candidate))):

#         equal = int(query[slot]==candidate[slot])

#         f[f"slot_{slot}"] = equal

#         if equal:
#             matches += 1
#             rarity_score += rarity(slot,query[slot])

#     f["matches"] = matches
#     f["rarity"] = rarity_score

#     return f

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.neighbors import NearestNeighbors

# from sklearn.feature_extraction.text import TfidfVectorizer

# vectorizer = TfidfVectorizer(
#     stop_words="english",
#     ngram_range=(1,3),
#     min_df=2,
#     max_df=0.9,
#     binary=True,
#     sublinear_tf=True
# )

# X_train = vectorizer.fit_transform(train_df["Reviews"])
# X_test = vectorizer.transform(test_df["Reviews"])

# nn = NearestNeighbors(
#     metric="cosine",
#     algorithm="brute",
#     n_neighbors=100
# )

# nn.fit(X_train)

# distances, indices = nn.kneighbors(X_test)

# def rerank(query_code, candidates, cosine_scores):

#     scored = []

#     for idx, dist in zip(candidates, cosine_scores):

#         cand_code = train_encoded.iloc[idx]

#         feat = features(query_code, cand_code, 1 - dist)

#         score = (
#             5 * feat["cosine"] +
#             2 * feat["matches"] +
#             0.5 * feat["rarity"] -
#             0.25 * feat["length_diff"]
#         )

#         scored.append((score, idx))

#     scored.sort(reverse=True)

#     return [idx for score, idx in scored]

# final_predictions = []

# for i in tqdm(range(len(test_df))):

#     ranked = rerank(
#         test_encoded.iloc[i],
#         indices[i],
#         distances[i]
#     )

#     final_predictions.append([int(x) for x in ranked[:10]])
    
# submission = pd.DataFrame({
#     "Index": test_df["Index"],
#     "Index_list": [str(x) for x in final_predictions]
# })

# submission.to_csv("submission11.csv", index=False)

# qid = 0

# print("Original")

# for idx, dist in zip(indices[qid][:10], distances[qid][:10]):
#     print(idx, round(1 - dist, 4))

# print("\nReranked")

# reranked = rerank(
#     test_encoded.iloc[qid],
#     indices[qid],
#     distances[qid]
# )

# for idx in reranked[:10]:
#     print(idx)

import re

def split_review(review):
    return [
        s.strip()
        for s in re.split(r"\.\s*", review)
        if s.strip()
    ]
def extract_course(review):
    s = split_review(review)
    return s[1]          

train_courses = train_df["Reviews"].apply(extract_course)

# from collections import Counter

# unique_course_counts = Counter()

# for i in range(len(test_df)):
#     courses = train_courses.iloc[indices[i][:10]].tolist()
#     unique_course_counts[len(set(courses))] += 1

# print(unique_course_counts)

# for i in range(len(test_df)):

#     courses = train_courses.iloc[indices[i][:10]].tolist()

#     if len(set(courses)) > 1:

#         print("="*80)
#         print("Query:", i)

#         for c in Counter(courses).most_common():
#             print(c)

#         print()
        
# qid = 0

# print("Rank   Cosine")

# for rank, dist in enumerate(distances[qid][:20], 1):
#     print(rank, round(1 - dist, 6))
    
# from collections import Counter

# for qid in range(5):

#     courses = train_courses.iloc[indices[qid][:100]]

#     print("="*80)
#     print("Query", qid)

#     print(Counter(courses).most_common(10))
    
# majority_changes = 0

# for i in range(len(test_df)):

#     top1 = train_courses.iloc[indices[i][0]]

#     majority = Counter(
#         train_courses.iloc[indices[i][:10]]
#     ).most_common(1)[0][0]

#     if top1 != majority:
#         majority_changes += 1

# print("Majority changed:", majority_changes)

# qid = 0

# print("="*100)
# print("TEST REVIEW")
# print(test_df.iloc[qid]["Reviews"])

# print("\nTOP 20")

# for rank, (idx, dist) in enumerate(zip(indices[qid][:20], distances[qid][:20]), 1):

#     print("="*80)

#     print(
#         f"Rank={rank} "
#         f"Cosine={1-dist:.6f} "
#         f"TrainIndex={idx}"
#     )

#     print(train_df.iloc[idx]["Reviews"])


from collections import Counter

import re


split_reviews = train_df["Reviews"].map(split_review)

max_slots = max(len(x) for x in split_reviews)

template_maps = []

for slot in range(max_slots):
    sentences = []

    for review in split_reviews:
        if slot < len(review):
            sentences.append(review[slot])

    cnt = Counter(sentences)

    mp = {sentence: i for i, sentence in enumerate(sorted(cnt))}
    template_maps.append(mp)


def encode(review):
    s = split_review(review)

    ids = []

    for i, sent in enumerate(s):
        ids.append(template_maps[i].get(sent, -1))

    return tuple(ids)


train_df["template"] = train_df["Reviews"].map(encode)

from collections import defaultdict

course_templates = defaultdict(Counter)

for review in train_df["Reviews"]:
    course = split_review(review)[1]
    temp = encode(review)
    course_templates[course][temp] += 1
    
for course, counter in list(course_templates.items())[:10]:

    print("=" * 80)
    print(course)

    print("Unique combinations:", len(counter))

    print("Most common:")

    for comb, freq in counter.most_common(10):
        print(freq, comb)


duplicates = Counter(train_df["template"])

print("Unique template combinations:", len(duplicates))
print("Maximum frequency:", max(duplicates.values()))

print()

print("Top 20")

for k, v in duplicates.most_common(20):
    print(v, k)
    
first_course = next(iter(course_templates))

reference = set(course_templates[first_course].keys())

same = True

for course, counter in course_templates.items():
    if set(counter.keys()) != reference:
        same = False
        print(course, "DIFFERENT")

print("All identical:", same)

from collections import Counter

pattern_count = Counter()

for review in train_df["Reviews"]:
    s = split_review(review)

    # Ignore slot 1 (course-specific sentence)
    pattern = (
    s[0] if len(s) > 0 else None,
    s[2] if len(s) > 2 else None,
    s[3] if len(s) > 3 else None,
    s[4] if len(s) > 4 else None,
    s[5] if len(s) > 5 else None,
    s[6] if len(s) > 6 else None,
)

    pattern_count[pattern] += 1
    
print("Maximum frequency:", max(duplicates.values()))

print("Unique patterns:", len(pattern_count))

print("\nTop 20 patterns:")
for pat, freq in pattern_count.most_common(20):
    print(freq)