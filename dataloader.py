""" train and test dataset


"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from collections import Counter

class data_reader(Dataset):
    def __init__(self, path, transform=None):
        #if transform is given, we transoform data using
        csv_data = pd.read_csv(path)
        data = csv_data.values
        self.names = data[:, 1]
        self.names = [os.path.join('../cancer_data/new_data', i+'.jpg') for i in self.names]
        self.labs = data[:, -1]
        print('data is {}'.format(path))
        print(Counter(self.labs))
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        img = Image.open(self.names[index]).convert('RGB')
        label = self.labs[index]
        if self.transform:
            image = self.transform(img)
            image = np.asarray(image)
        else:
            image = np.asarray(img)
        return image, np.asarray([label])