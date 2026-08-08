import re
from collections import defaultdict
import numpy as np;
import pandas as pd;
import seaborn as sns;
import matplotlib.pyplot as plt;
from tqdm import tqdm;

train_df = pd.read_csv("train.csv");
test_df = pd.read_csv("test.csv");

import re

def extract_course(review):
    s = split_review(review)
    return s[1]          

train_courses = train_df["Reviews"].apply(extract_course)