import pandas as pd
import numpy as np

df = pd.read_csv("sorted.csv")
num_of_seq = len(pd.unique(df["subject_id"]))

print(num_of_seq)