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

def load_chronic_diseases():
    # load chronic disease indicator
    ind9 = pd.read_csv("cci2015.csv")
    ind10 = pd.read_csv("CCIR_v2023-1.csv")

    ind9_code = {}
    ind10_code = {}
    print(ind9)
    # with the indicator map the chronic diseases and their titles into the list
    for index, row in ind9.iterrows():
        if row['CATEGORY DESCRIPTION'] == "'1'":
            ind9_code[row['ICD-9-CM CODE']] = row['ICD-9-CM CODE DESCRIPTION']

    for index, row in ind10.iterrows():
        if row['CHRONIC INDICATOR'] == 1:
            ind10_code[row['ICD-10-CM CODE']] = row['ICD-10-CM CODE DESCRIPTION']

    with open("icd9.csv", "a+") as f:
        f.write("code"+"	"+"title"+"\n")
        for i in ind9_code:
            f.write(str(i).replace(' ', '')[1:-1]+"	"+str(ind9_code[i])+"\n")

    with open("icd10.csv", "a+") as f:
        f.write("code"+"	"+"title"+"\n")
        for i in ind10_code:
            f.write(str(i).replace(' ', '')[1:-1]+"	"+str(ind10_code[i])+"\n") 

    return ind9_code, ind10_code


record_dict = load_data()
load_chronic_diseases()

df1 = pd.read_csv("icd9.csv", delimiter="	")
df2 = pd.read_csv("icd10.csv", delimiter="	")

codes9 = df1["code"].to_list()
codes10 = df2["code"].to_list()


new_dict = {}
for record in record_dict:
    search = len(record_dict[record][-7])
    i = 0
    align = 0
    while i < search:
        version = record_dict[record][-6][i-align]
        code = record_dict[record][-7][i-align]
        match version:
            case 9:
                if code not in codes9:
                    record_dict[record][-6].pop(i-align)
                    record_dict[record][-7].pop(i-align)
                    record_dict[record][-8].pop(i-align)
                    align += 1
            case 10:
                if code not in codes9:
                    record_dict[record][-6].pop(i-align)
                    record_dict[record][-7].pop(i-align)
                    record_dict[record][-8].pop(i-align)
                    align += 1
        i += 1
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
    df_dict["icd_code"].append(new_dict[record][-7])
    df_dict["icd_version"].append(new_dict[record][-6])

df = pd.DataFrame(df_dict)


df = df[df['icd_code'].astype(bool)]
counts = df['subject_id'].value_counts()
df_filtered = df[df['subject_id'].isin(counts[counts >= 2].index)]

print(df_filtered)

df_filtered.to_csv("filtered.csv", index=False)
                    






