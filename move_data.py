import pandas as pd
import os
import shutil
import tqdm

p = '../siimisic_data/filter.csv'
src = '../siimisic_data/jpeg_data/train'
tgt = '../siimisic_data/new_train'
# os.mkdir(tgt)
csv_reader = pd.read_csv(p)
data = csv_reader.values

for item in tqdm.tqdm(data):
    name = item[1]+'.jpg'
    shutil.move(os.path.join(src, name), os.path.join(tgt, name))