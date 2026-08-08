import pandas as pd
import re

train = pd.read_csv("train.csv")

def split(review):
    s = re.split(r'(?<=[.!?])\s+', review.strip())
    while len(s)<6:
        s.append("")
    return s[:6]

out = open("cartesian_check.txt","w",encoding="utf8")

for course,group in train.groupby("Course"):

    s3=set()
    s4=set()
    s5=set()
    s6=set()

    combos=set()

    for review in group["Reviews"]:

        s=split(review)

        s3.add(s[2])
        s4.add(s[3])
        s5.add(s[4])
        s6.add(s[5])

        combos.add((s[2],s[3],s[4],s[5]))

    possible=len(s3)*len(s4)*len(s5)*len(s6)

    out.write("="*80+"\n")
    out.write(course+"\n")

    out.write(f"S3 : {len(s3)}\n")
    out.write(f"S4 : {len(s4)}\n")
    out.write(f"S5 : {len(s5)}\n")
    out.write(f"S6 : {len(s6)}\n")

    out.write(f"Possible : {possible}\n")
    out.write(f"Observed : {len(combos)}\n")
    out.write(f"Coverage : {len(combos)/possible:.3f}\n\n")

out.close()

print("Done")