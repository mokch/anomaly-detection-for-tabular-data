from dataloader import get_loader
from models.goad import GOAD_netC1, GOAD_netC5, GOAD_tc_loss
import torch
import numpy as np
import os
import torch.nn.functional as F
import torch.nn as nn

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2

class trainer():
    def __init__(self, config):
        self.__dict__.update(trainer.DEFAULTS, **config)
        self.train_loader, self.valid_loader, self.test_loader, self.model =  self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self, config):
        if config.model == 'GOAD':
            #hyper parameters (n_rots, n_epoch, d_out, ndf, dataset_name)
            # (64, 25, 64, 32) for kdd
            # (256, 25, 128, 128) for kddrev
            # (256, 1, 32, 8) for thyroid
            # (256, 1, 32, 8) for arrhythmia
            self.n_rots, self.n_epoch, self.d_out, self.ndf = (256, 25, 128, 128)
                        
            if config.dataset_name == "thyroid" or config.dataset_name == "arrhythmia":
                self.model = GOAD_netC1(self.d_out, self.ndf, self.n_rots).cuda()
                self.n_rots, self.n_epoch, self.d_out, self.ndf = (256, 1, 32, 8)
            else:
                self.model = GOAD_netC5(self.d_out, self.ndf, self.n_rots).cuda()

            self.model.weights_init(self.model)
            self.optimizerC = torch.optim.Adam(self.model.parameters(), lr=config.lr, betas=(0.5, 0.999))
            self.transform = 
            self.loss = GOAD_tc_loss()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def train(model, data, args):
        epochs = args.epochs

        for i in range(epochs):

