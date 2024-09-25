import pandas as pd

df = pd.read_csv("test.csv")

grouped = df.groupby('subject_id')

merged_df = grouped.agg({
    'anchor_age': lambda x : list(map(str, x)),
    'icd_code': lambda x: list(map(str, x)),
    'icd_version': lambda x: list(map(str, x)),
    'hadm_id': lambda x: list(map(str, x)),
    'admit_time': lambda x: list(map(str, x)),
    'disch_time': lambda x: ', '.join(x),
    'anchor_year': lambda x: list(map(str, x))
}).reset_index()

# for col in df.columns:
#     if col not in ['icd_code', 'icd_version']:
#         merged_df[col] = grouped[col].first()

# merged_df = merged_df.reset_index()

print(merged_df)
merged_df.to_csv("sequences.csv")