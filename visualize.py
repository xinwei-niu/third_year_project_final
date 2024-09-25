import matplotlib.pyplot as plt
import numpy as np
import json 
from random import random
from matplotlib.ticker import MultipleLocator
from datetime import datetime
data = []
with open("data.jsonl") as f:
    for line in f:
        objs = [i for i in json.loads(line)["occurance"]]
        objs.sort(key=lambda x: datetime.strptime(x["admit_time"], '%Y-%m-%d %H:%M:%S'))
        data.append(objs)
pn = []
for i in data:
    points = []
    
    for j in i:
        anchor_age, anchor_year = j["anchor_age"], np.datetime64(str(j["anchor_year"])+"-01-01 00:00:00")
        points.append((j["icd_code"], (np.float32(np.timedelta64((np.datetime64(j["admit_time"])-anchor_year), 'D'))+365.25*(anchor_age-18))/365.25) )
    pn.append(points)
# print(data)

fig, ax = plt.subplots()
count = 0
for point_set in pn:
    print(count)
    
    rand_col = (random(), random(), random())
    y_vals, x_vals = zip(*point_set)
    # ax.plot(x_vals,y_vals, "-o", markersize=0.65,color = rand_col, alpha=0.5, linewidth=0.3)
    ax.scatter(x_vals,y_vals,color = rand_col, s=0.5)
    count+=1
ax.set_xlim([0, 100])  # Set x-axis limits
ax.set_ylim([0, 1858]) 
plt.show()
