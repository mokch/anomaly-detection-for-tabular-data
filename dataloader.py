import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from scipy import io
from sklearn.model_selection import train_test_split

class ODDSLoader(object):
    def __init__(self, data_path, data_name, mode="train", seed=2023, test_size=0.5):
        self.mode = mode
        self.name = data_name
        self.scaler = StandardScaler()
        self.random_state = seed

        data = io.loadmat(os.path.join(data_path, f'{data_name}.mat'))

        X = data['X']
        y = np.array(data['y']).reshape(-1)
        normal = X[y==0]
        abnormal = X[y==1]

        train_X_, test_X = train_test_split(normal, test_size=test_size, random_state=seed)
        train_X, valid_X = train_test_split(train_X_, test_size=0.1, random_state=seed)
        test_X = np.concatenate((test_X, abnormal), axis = 0)
        test_y = np.concatenate((np.zeros(test_X.shape[0]-abnormal.shape[0]), np.ones(abnormal.shape[0])))

        self.scaler.fit(train_X)
        self.train = self.scaler.transform(train_X)
        self.valid = self.scaler.transform(valid_X)
        self.test = self.scaler.transform(test_X)
        self.test_labels = test_y

        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print("valid:", self.valid.shape)


    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        elif (self.mode == 'valid'):
            return self.valid.shape[0]
        elif (self.mode == 'test'):
            return self.test.shape[0]
        else:
            return self.test.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.zeros(self.train.shape[0])[index]
        elif (self.mode == 'valid'):
            return np.float32(self.valid[index]), np.zeros(self.valid.shape[0])[index]
        elif (self.mode == 'test'):
            return np.float32(self.test[index]), np.float32(self.test_labels[index])
        else:
            return 0
    def get_size(self):
        return self.train.shape[1]

def get_loader(data_path, batch_size, mode='train', dataset='tabular', data_name='thyroid'):

    dataset = ODDSLoader(data_path, data_name, mode)

    if mode == 'train':
        shuffle = True
    else:
        shuffle = False

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return data_loader