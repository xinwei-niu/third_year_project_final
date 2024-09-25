import json

pad_index = 0
anchored_age = 100
lowest_bound = 0
with open("data.jsonl", 'r') as f:
    for line in f:
        objs = [i for i in json.loads(line)["occurance"]]
        anchor_age, anchor_year = objs[0]["anchor_age"], objs[0]["anchor_year"]
        # self.type_data.append([i["admit_time"] for i in json.loads(line)["occurance"]])
        # batch_type_seq = np.array([i["icd_code"] for i in objs + [pad_index] * (238 - len(i["icd_code"] for i in objs)) ], dtype=np.int64)
        # batch_type_seq = np.array([[i["icd_code"] for i in objs] + [pad_index] * (238 - len([i["icd_code"] for i in objs])) ], dtype=np.int64)
        
        # batch_time_seq = np.array([[i["admit_time"] for i in objs] + [pad_index] * (238 - len([i["admit_time"] for i in objs])) ], dtype=np.datetime64)
        # #self.time_data.append(([i["icd_code"] for i in json.loads(line)["occurance"]]))
        if anchored_age > anchor_age:
            anchored_age = anchor_age
        if lowest_bound < anchor_age:
            lowest_bound = anchor_age
    print(anchored_age)
    print(lowest_bound)
        # self.data.append([batch_type_seq,batch_time_seq])