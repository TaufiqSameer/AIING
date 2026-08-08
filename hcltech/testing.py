import pandas as pd
import re

train = pd.read_csv("train.csv")


def split(review):
    s = re.split(r'(?<=[.!?])\s+', review.strip())
    while len(s) < 6:
        s.append("")
    return s[:6]


with open("pairwise_analysis.txt", "w", encoding="utf-8") as f:

    for course, group in train.groupby("Course"):

        f.write("=" * 100 + "\n")
        f.write(course + "\n")
        f.write("=" * 100 + "\n\n")

        s3 = [split(r)[2] for r in group["Reviews"]]
        s4 = [split(r)[3] for r in group["Reviews"]]
        s5 = [split(r)[4] for r in group["Reviews"]]
        s6 = [split(r)[5] for r in group["Reviews"]]

        df = pd.DataFrame({
            "S3": s3,
            "S4": s4,
            "S5": s5,
            "S6": s6
        })

        pairs = [
            ("S3", "S4"),
            ("S3", "S5"),
            ("S3", "S6"),
            ("S4", "S5"),
            ("S4", "S6"),
            ("S5", "S6"),
        ]

        for a, b in pairs:

            f.write(f"{a} vs {b}\n")
            f.write("-" * 40 + "\n")

            stats = df.groupby([a, b]).size().describe()

            f.write(stats.to_string())
            f.write("\n\n")

        f.write("\n\n")

print("Saved analysis to pairwise_analysis.txt")