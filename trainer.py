from dataloader import get_loader
from models.goad import GOAD, GOAD_loss
from models.pca_goad import PCA_GOAD, GOAD_loss
import torch
import numpy as np
import os
import torch.nn.functional as F
import torch.nn as nn
import time
from metrics import accuracy_score
import pandas as pd

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, data_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.data_name = data_name


    def __call__(self, valid_loss, model, path):
        score = valid_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(valid_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(valid_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.data_name) + '_checkpoint.pth'))
        self.val_loss_min = val_loss

class trainer():
    def __init__(self, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.epochs = config.epochs
        self.model_save_path = config.model_save_path
        self.data_name = config.data_name
        self.build_model(config)
        self.model_name = config.model
        self.result = pd.DataFrame()

    def build_model(self, config):
        if config.model == 'GOAD':
            goad = GOAD(config)
            self.model = goad.build_model()
            self.train_loader, self.valid_loader, self.test_loader, self.trans_matrix = goad.prepare_data()
            self.ndf = goad.ndf
            self.n_rots = goad.n_rots
            self.eps = 0

            self.optimizerC = torch.optim.Adam(self.model.parameters(), lr=config.lr, betas=(0.5, 0.999))
            self.loss = GOAD_loss()
        if config.model == 'PCA_GOAD':
            goad = PCA_GOAD(config)
            self.model = goad.build_model()
            self.train_loader, self.valid_loader, self.test_loader, self.trans_matrix = goad.prepare_data()
            self.ndf = goad.ndf
            self.n_rots = goad.n_rots
            self.eps = 0

            self.optimizerC = torch.optim.Adam(self.model.parameters(), lr=config.lr, betas=(0.5, 0.999))
            self.loss = GOAD_loss()            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def validate(self, dataloader):
        self.model.eval()
        loss_list = []
        with torch.no_grad():
            for i, (data, label) in enumerate(dataloader):

                self.optimizer.zero_grad()
                X = data.to(self.device)
                label = label.to(self.device)
                latent, pred = self.model(X)
                loss =self.loss.tc_loss(latent, pred, label)
                loss_list.append(loss.item())
         
        return np.average(loss_list)

    def train(self):

        print("TRAIN")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, data_name=self.data_name)
        train_steps = len(self.train_loader)

        for epoch in range(self.epochs):
            iter_count = 0
            loss_list = []

            epoch_time = time.time()
            self.model.train()
            sum_latent = torch.zeros((self.ndf, self.n_rots)).to(self.device)

            for i, (data, label) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1

                X = data.to(self.device)
                label = label.to(self.device)
                latent, pred = self.model(X)

                sum_latent = sum_latent + latent.mean(axis=0) 
                latent = latent.permute(0, 2, 1)


                loss =self.loss.tc_loss(latent, pred, label)

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss_list.append(loss.item())
                loss.backward()
                self.optimizer.step()

            means = sum_latent.t() / len(self.train_loader)
            self.means = means.unsqueeze(0)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss_list)

            valid_loss = self.validate(self.valid_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, valid_loss))
            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def test(self):
        self.model.load_state_dict(torch.load(os.path.join(str(self.model_save_path), str(self.data_name) + '_checkpoint.pth')))
        self.model.eval()
        with torch.no_grad():
            val_probs_rots = np.zeros((len(self.test_loader), self.n_rots))
            ab_num = 0
            labels = []
            for i, (data, label) in enumerate(self.test_loader):
                X = data.to(self.device)
                latent, pred = self.model(X)
                latent = latent.permute(0, 2, 1)
                diffs = ((latent.unsqueeze(2) - self.means) ** 2).sum(-1)

                diffs_eps = self.eps * torch.ones_like(diffs)
                diffs = torch.max(diffs, diffs_eps)
                logp_sz = torch.nn.functional.log_softmax(-diffs, dim=2)
                ab_num += label.item()
                labels.append(label.item())
                val_probs_rots[i] = -torch.diagonal(logp_sz, 0, 1, 2).cpu().data.numpy()

            val_probs_rots = val_probs_rots.sum(1)
            ratio = (len(self.test_loader)-ab_num)/len(self.test_loader)
            precision, recall, f_score, support, auroc = accuracy_score(val_probs_rots, labels, ratio)
            print("f1_score: ", f_score, ", AUROC: ", auroc)
            result = {'data':self.data_name,'model':self.model_name,'n_rots':self.n_rots,'precision':precision,'recall':recall,'f1_score':f_score,'support':support,'AUROC':auroc}

        self.result = self.result.append(pd.DataFrame.from_dict([result]))

    