import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from dataloader import ODDSLoader
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import io
import os
from sklearn.decomposition import PCA
import random

class GOAD_loss(torch.nn.Module):
    def __init__(self, alpha = 0.1, margin = 1, device = 'cuda'):
        super(GOAD_loss, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.alpha = alpha
        self.margin = margin
        self.device = device

    def tc_loss(self, latent, pred, real):        
        ce_loss = self.ce_loss(pred, real)
        means = latent.mean(0).unsqueeze(0)
        res = ((latent.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
        pos = torch.diagonal(res, dim1=1, dim2=2)
        offset = torch.diagflat(torch.ones(latent.size(1))).unsqueeze(0).to(self.device) * 1e6
        neg = (res + offset).min(-1)[0]
        tc_loss = torch.clamp(pos + self.margin - neg, min=0).mean()
        loss = self.alpha*tc_loss + ce_loss
        return loss


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Conv') != -1:
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Linear') != -1:
        init.eye_(m.weight)
    elif classname.find('Emb') != -1:
        init.normal(m.weight, mean=0, std=0.01)

class ODDSpcaLoader(object):
    def __init__(self, data_path, data_name, pca_list, mode="train", seed=2023, test_size=0.5):
        # super().__len__(self)
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

        self.train = np.stack([pca.transform(self.train) for pca in pca_list], 2) # shape: [n_samples, trans_dim, n_trans]
        self.valid = np.stack([pca.transform(self.valid) for pca in pca_list], 2)
        self.test = np.stack([pca.transform(self.test) for pca in pca_list], 2)

        classification_labels = np.arange(len(pca_list))
        self.train_labels = classification_labels.repeat(self.train.shape[0], 0).reshape(self.train.shape[0],-1)
        self.valid_labels = classification_labels.repeat(self.valid.shape[0], 0).reshape(self.valid.shape[0],-1)


    def __getitem__(self, idx):
        if self.mode == "train":
            return np.float32(self.train[idx]), np.int64(self.train_labels[idx])
        elif self.mode == 'valid':
            return np.float32(self.valid[idx]), np.int64(self.valid_labels[idx])
        elif self.mode == 'test':
            return np.float32(self.test[idx]), np.int64(self.test_labels[idx])
        else:
            return 0

    def get_size(self):
        return self.train.shape[1]

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        elif (self.mode == 'valid'):
            return self.valid.shape[0]
        elif (self.mode == 'test'):
            return self.test.shape[0]
        else:
            return self.test.shape[0]

def get_loader(data_path, batch_size, pca_list, dataset='tabular', data_name='thyroid', mode='train'):

    dataset = ODDSpcaLoader(data_path, data_name, pca_list, mode)

    if mode == 'train':
        shuffle = True
    else:
        shuffle = False

    if mode == 'test':
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=shuffle, num_workers=0)
    else:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return data_loader



class PCA_GOAD(object):
    def __init__(self, config):
        super(PCA_GOAD, self).__init__()
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.data_name = config.data_name
        self.dataset = config.dataset
        dataset = ODDSLoader(self.data_path, self.data_name, mode='train')
        self.n_features = dataset.get_size()
        self.d_out = self.n_features//2

        if config.data_name == 'kdd':
            self.n_rots, self.n_epoch, self.ndf = (64, 1000, 32)
            self.model = GOAD_netC5(self.d_out, self.ndf, self.n_rots)

        elif config.data_name == 'kddrev':
            self.n_rots, self.n_epoch, self.ndf = (256, 1000, 128)
            self.model = GOAD_netC5(self.d_out, self.ndf, self.n_rots)

        elif config.data_name == "thyroid" or config.data_name == "arrhythmia":
            self.n_rots, self.n_epoch, self.ndf = (256, 1000, 8)
            self.model = GOAD_netC1(self.d_out, self.ndf, self.n_rots)

        else:
            self.n_rots, self.n_epoch, self.ndf = (256, 1000, 128)
            self.model = GOAD_netC5(self.d_out, self.ndf, self.n_rots)

        weights_init(self.model)
        self.optimizerC = optim.Adam(self.model.parameters(), lr=config.lr, betas=(0.5, 0.999))
        



    def prepare_data(self):

        dataset = ODDSLoader(self.data_path, self.data_name, mode='train')
        self.pca_list = []
        for i in range(self.n_rots):
            pca = PCA(n_components=self.n_features//2)
            # idx = random.sample(list(np.arange(n_train)), n_train//args.n_rots)
            idx = random.sample(list(np.arange(dataset.train.shape[0])), 100)
            self.pca_list.append(pca.fit(dataset.train[idx]))

        train_loader = get_loader(self.data_path, self.batch_size, self.pca_list, self.dataset, self.data_name, mode='train')
        valid_loader = get_loader(self.data_path, self.batch_size, self.pca_list, self.dataset, self.data_name, mode='valid')
        test_loader = get_loader(self.data_path, self.batch_size, self.pca_list, self.dataset, self.data_name, mode='test')
        return train_loader, valid_loader, test_loader, self.pca_list

    def build_model(self):
        return self.model

    def build_optimizer(self):
        return self.optimizerC

    def trans_matrix(self):
        return self.trans_matrix

class GOAD_netC5(nn.Module):
    def __init__(self, d, ndf, nc):
        super(GOAD_netC5, self).__init__()
        self.trunk = nn.Sequential(
        nn.Conv1d(d, ndf, kernel_size=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv1d(ndf, ndf, kernel_size=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv1d(ndf, ndf, kernel_size=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv1d(ndf, ndf, kernel_size=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv1d(ndf, ndf, kernel_size=1, bias=False),
        )
        self.head = nn.Sequential(
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv1d(ndf, nc, kernel_size=1, bias=True),
        )

    def forward(self, input):
        tc = self.trunk(input)
        ce = self.head(tc)
        return tc, ce


class GOAD_netC1(nn.Module):
    def __init__(self, d, ndf, nc):
        super(GOAD_netC1, self).__init__()
        self.trunk = nn.Sequential(
        nn.Conv1d(d, ndf, kernel_size=1, bias=False),
        )
        self.head = nn.Sequential(
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv1d(ndf, nc, kernel_size=1, bias=True),
        )

    def forward(self, input):
        tc = self.trunk(input)
        ce = self.head(tc)
        return tc, ce