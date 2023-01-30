import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from dataloader import ODDSLoader
from torch.utils.data import DataLoader

def GOAD_tc_loss(zs, m):
    means = zs.mean(0).unsqueeze(0)
    res = ((zs.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
    pos = torch.diagonal(res, dim1=1, dim2=2)
    offset = torch.diagflat(torch.ones(zs.size(1))).unsqueeze(0).cuda() * 1e6
    neg = (res + offset).min(-1)[0]
    loss = torch.clamp(pos + m - neg, min=0).mean()
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

class ODDStransLoader(ODDSLoader):
    def __init__(self, data_path, data_name, mode="train", seed=2023, test_size=0.5):
        super().__init__(data_path, data_name, mode, seed, test_size)
        super().__len__(self)

        self.trans = np.random.randn(self.n_trans, self.n_features, self.trans_dim)

        self.train_trans = np.stack([self.train.dot(rot) for rot in self.affine_weights], 2) # shape: [n_samples, trans_dim, n_trans]
        self.valid_trans = np.stack([self.valid.dot(rot) for rot in self.affine_weights], 2) 
        self.test_trans = np.stack([self.test.dot(rot) for rot in self.affine_weights], 2)

        def __getitem__(self, index):
            if self.mode == "train":
                return np.float32(self.train), np.zeros(self.train.shape[0])
            elif (self.mode == 'valid'):
                return np.float32(self.valid), np.zeros(self.valid.shape[0])
            elif (self.mode == 'test'):
                return np.float32(self.test), np.float32(self.test_labels)
            else:
                return 0

        def trans_matrix(self):
            return self.trans

def get_loader(data_path, batch_size, mode='train', dataset='tabular', data_name='thyroid'):

    dataset = ODDStransLoader(data_path, data_name, mode)

    if mode == 'train':
        shuffle = True
    else:
        shuffle = False

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return data_loader



class GOAD(object):
    def __init__(self, config):
        super(GOAD, self).__init__()

        if config.data_name == 'kdd':
            self.n_rots, self.n_epoch, self.d_out, self.ndf = (64, 25, 64, 32)
            self.model = GOAD_netC5(self.d_out, self.ndf, self.n_rots).to(config.device)

        elif config.data_name == 'kddrev':
            self.n_rots, self.n_epoch, self.d_out, self.ndf = (256, 25, 128, 128)
            self.model = GOAD_netC5(self.d_out, self.ndf, self.n_rots).to(config.device)

        elif config.data_name == "thyroid" or config.data_name == "arrhythmia":
            self.n_rots, self.n_epoch, self.d_out, self.ndf = (256, 1, 32, 8)
            self.model = GOAD_netC1(self.d_out, self.ndf, self.n_rots).to(config.device)

        else:
            self.n_rots, self.n_epoch, self.d_out, self.ndf = (256, 25, 128, 128)
            self.model = GOAD_netC5(self.d_out, self.ndf, self.n_rots).to(config.device)

        weights_init(self.model)
        self.optimizerC = optim.Adam(self.model.parameters(), lr=config.lr, betas=(0.5, 0.999))

    def prepare_data(data_path, batch_size, data_name):
        train_loader = get_loader(data_path, batch_size=batch_size, mode='train', data_name=data_name)
        valid_loader = get_loader(data_path, batch_size=batch_size, mode='valid', data_name=data_name)
        test_loader = get_loader(data_path, batch_size=batch_size, mode='test', data_name=data_name)
        return train_loader, valid_loader, test_loader

    def build_model(self):
        return self.model

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