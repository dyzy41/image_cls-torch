import os
import shutil
import random
import tqdm

p = '../data/jpeg_data/train'
p2 = '../data/jpeg_data/val'
imgs = os.listdir(p)
random.shuffle(imgs)
val_imgs = imgs[:int(len(imgs)*0.2)]
for item in tqdm.tqdm(val_imgs):
    shutil.move(os.path.join(p, item), os.path.join(p2, item))