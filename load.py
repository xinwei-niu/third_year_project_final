import json
import pandas as pd
import ast
df = pd.read_csv("sequences.csv")

df2 = pd.read_csv("test.csv")
num_of_seq = len(pd.unique(df2["icd_code"]))
print(pd.unique(df2["icd_code"]).__class__)
print(num_of_seq)
hash = {}
for i in range(num_of_seq):
    hash[pd.unique(df2["icd_code"])[i]] = i+1

sequences = []

longest = 0    
for index, row in df.iterrows():
    # print(row)
    temp_dict = {}
    temp_dict["subject_id"] = int(df.loc[index, "subject_id"])

    temp_dict["occurance"] = []
    # if longest < len(ast.literal_eval(row["icd_code"])):
    #     longest = len(ast.literal_eval(row["icd_code"]))
    #     print(longest)
    for i in range(len(ast.literal_eval(row["icd_code"]))):
        
        # print(ast.literal_eval(row["icd_code"]))
        temp_dict["occurance"].append({"icd_code":hash[str(ast.literal_eval(row["icd_code"])[i])], 
                                       "icd_version":int(ast.literal_eval(row["icd_version"])[i]), 
                                       "hadm_id":int(ast.literal_eval(row["hadm_id"])[i]), 
                                       "admit_time":str(ast.literal_eval(row["admit_time"])[i]),
                                       "anchor_year":int(ast.literal_eval(row["anchor_year"])[i]), 
                                        "anchor_age":int(ast.literal_eval(row["anchor_age"])[i])})
    sequences.append(temp_dict)



# print(sequences)

with open('data.jsonl', 'a+') as f:
    for entry in sequences:
        json.dump(entry, f)
        f.write('\n')

print(longest)