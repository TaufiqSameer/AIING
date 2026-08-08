from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import re

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

def anonymize_first_sentence(review, course):
    sentences = re.split(r'(?<=[.!?])\s+', review)

    if sentences:
        sentences[0] = re.sub(
            re.escape(course),
            "this course",
            sentences[0],
            flags=re.IGNORECASE,
        )

    return " ".join(sentences)

train_df["AnonReview"] = train_df.apply(
    lambda row: anonymize_first_sentence(
        row["Reviews"],
        row["Course"]
    ),
    axis=1
)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

train_embeddings = model.encode(
    train_df["AnonReview"].tolist(),
    batch_size=128,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

test_embeddings = model.encode(
    test_df["Reviews"].tolist(),
    batch_size=128,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

nn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=10,
)

nn.fit(train_embeddings)

distances, indices = nn.kneighbors(test_embeddings)

submission = pd.DataFrame({
    "Index": test_df["Index"],
    "Index_list": [
        str(train_df.iloc[idx]["Index"].tolist())
        for idx in indices
    ]
})

submission.to_csv(
    "submission_minilm.csv",
    index=False,
)