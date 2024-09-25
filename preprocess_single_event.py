import pandas as pd
def load_data():
    # load MIMIC data
    df = pd.read_csv("admissions.csv")
    df3 = pd.read_csv("diagnoses_icd.csv")
    df4 = pd.read_csv("patients.csv")
    df = pd.merge(df, df3[['hadm_id', 'icd_code', 'seq_num', 'icd_version']], on='hadm_id',suffixes=('', ''))
    print(df)
    df = pd.merge(df, df4[["subject_id", "gender","anchor_age","anchor_year","anchor_year_group","dod"]], on='subject_id',suffixes=('', ''))
    record_dict = {}

    for index, row in df.iterrows():
        if row["hadm_id"] not in record_dict:
            record_dict[row["hadm_id"]] = [row["subject_id"], row["admittime"], row["dischtime"], row["deathtime"], row["admission_type"],
                                            row["admit_provider_id"], row["admission_location"], row["discharge_location"], row["insurance"], 
                                            row["language"], row["marital_status"], row["race"], row["edregtime"],row["edouttime"],
                                            row["hospital_expire_flag"], [row["seq_num"]], [row["icd_code"]], [row["icd_version"]], row["gender"], row["anchor_age"], row["anchor_year"], row["anchor_year_group"], row["dod"]]

        else:
            record_dict[row["hadm_id"]][-8].append(row["seq_num"])
            record_dict[row["hadm_id"]][-7].append(row["icd_code"])
            record_dict[row["hadm_id"]][-6].append(row["icd_version"])


    return record_dict
record_dict = load_data()

new_dict = {}
for record in record_dict:

    search = len(record_dict[record][-7])
    
    if record_dict[record][-1] != []:
        new_dict[record] = record_dict[record]

df_dict = {"subject_id":[], "hadm_id":[], "gender":[], "anchor_age":[], "anchor_year":[], "admit_time":[], "disch_time":[], "icd_code":[], "icd_version":[]}

for record in new_dict:
    df_dict["subject_id"].append(new_dict[record][0])
    df_dict["hadm_id"].append(record)
    df_dict["admit_time"].append(new_dict[record][1])
    df_dict["disch_time"].append(new_dict[record][2])
    df_dict["gender"].append(new_dict[record][-5])
    df_dict["anchor_age"].append(new_dict[record][-4])
    df_dict["anchor_year"].append(new_dict[record][-3])
    df_dict["icd_code"].append(new_dict[record][-7][0][:3])
    df_dict["icd_version"].append(new_dict[record][-6][0])

df = pd.DataFrame(df_dict)

counts = df['subject_id'].value_counts()
df_filtered = df[df['subject_id'].isin(counts[counts >= 3].index)]

print(df_filtered)

df_filtered.to_csv("test.csv", index=False)
                    