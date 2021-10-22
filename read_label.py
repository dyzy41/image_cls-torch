import pandas as pd
from collections import Counter
import numpy as np
import random

p = '../siimisic_data/train.csv'

csv_reader = pd.read_csv(p)
data = csv_reader.values
data = data[data[:,-1].argsort()]
print(data[:, -1])

print(Counter(data[:, -1]))
need_data = list(data[-584*6:, :])
random.shuffle(need_data)
train_data = need_data[:int(len(need_data)*0.8)]
val_data = need_data[int(len(need_data)*0.8):]

train_data = pd.DataFrame(train_data)
train_data.to_csv("train.csv")
print('ok')

val_data = pd.DataFrame(val_data)
val_data.to_csv("val.csv")
print('ok')