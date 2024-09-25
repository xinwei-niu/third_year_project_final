import pandas as pd

df = pd.read_csv("sorted.csv")

df['admit_time'] = pd.to_datetime(df['admit_time'])
df['anchor_year'] = pd.to_datetime(df['anchor_year'].astype(str) + '-01-01')

# Calculate the age based on the difference between 'admit_time' and 'anchor_year'
df['current_age'] = df["anchor_age"]+(df['admit_time'].dt.year - df['anchor_year'].dt.year)


df.to_csv("shit.csv", index=False)
print(df)