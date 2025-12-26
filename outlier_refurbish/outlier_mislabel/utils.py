import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer, RobustScaler, MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
from run_detector import run_detector


class MyDataSet(Dataset):
    def __init__(self, data, index=None):
        self.data = data
        self.X = self.data[:, :-1]
        self.y = self.data[:, -1]

        if index is None:
            self.index = np.arange(len(self.X))
        else:
            self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        tensor_x = torch.tensor(self.X[index], dtype=torch.float)
        tensor_y = torch.tensor(self.y[index], dtype=torch.long)
        return index, tensor_x, tensor_y


class MyDataSetRecon(Dataset):
    def __init__(self, data, index=None):
        self.data = data[:, :-1]
        self.X = self.data

        if index is None:
            self.index = np.arange(len(self.X))
        else:
            self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        tensor_x = torch.tensor(self.X[index], dtype=torch.float)
        return tensor_x


class SoftLabelLoss(nn.Module):
    """
    compute the combined loss (hard label + soft label)
    """
    def __init__(self, alpha=1, temperature = 1, reduction = True):
        super(SoftLabelLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.temperature = temperature

    def forward(self, output, y_noisy, soft_label):
        T = self.temperature
        output_prob = F.log_softmax(output / T, dim=1)
        soft_label = soft_label / T
        if self.reduction == True:
            hard_label_loss =  F.cross_entropy(output, y_noisy, reduction='mean')
            soft_label_loss = F.kl_div(output_prob, soft_label, reduction='batchmean') * (T ** 2)
            combined_loss = self.alpha * hard_label_loss + (1-self.alpha) * soft_label_loss
        else:
            hard_label_loss = F.cross_entropy(output, y_noisy, reduction='none')
            soft_label_loss = F.kl_div(output_prob, soft_label, reduction='none').sum(dim=1) * (T ** 2)
            combined_loss = self.alpha * hard_label_loss + (1 - self.alpha) * soft_label_loss

        return combined_loss


class Contrastive_Loss(nn.Module):
    def __init__(self, center=None, eps = 1e-6):
        super(Contrastive_Loss, self).__init__()
        self.center = center
        self.eps = eps

    def forward(self, output, label):
        distance = torch.norm(output - self.center.unsqueeze(0), dim = 1)
        loss_contrastive = torch.mean((1 - label) * torch.pow(distance, 2) + label * (1 - torch.pow(distance, 2)))
        return loss_contrastive


class NegEntropy(object):
    """
    negative entropy penalty for over-confident prediction
    """
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))


class EMARecorder:
    '''
    Records the EMA loss and confidences for each sample
    '''

    def __init__(self, y_noisy):
        self.hist_loss_ema = torch.zeros((1, len(y_noisy))).cuda().detach() - 1
        self.hist_sim_ema = torch.zeros((1, len(y_noisy))).cuda().detach() - 1
        self.hist_entropy_ema = torch.zeros((1, len(y_noisy))).cuda().detach() - 1

    def record_loss_ema(self, idx, loss):
        self.hist_loss_ema[0, idx] = loss.clone().float().cuda().detach()

    def record_sim_ema(self, idx, sim):
        self.hist_sim_ema[0, idx] = sim.clone().float().cuda().detach()

    def record_entropy_ema(self, idx, entropy):
        self.hist_entropy_ema[0, idx] = entropy.clone().float().cuda().detach()

    def get_loss_ema(self, idx):
        return self.hist_loss_ema[0, idx]

    def get_sim_ema(self, idx):
        return self.hist_sim_ema[0, idx]

    def get_entropy_ema(self, idx):
        return self.hist_entropy_ema[0, idx]


def create_dataloader(args, X, y_real, y_noisy, soft_label):
    """

    """
    scaler = RobustScaler().fit(X)
    X_transformed = scaler.transform(X)

    # divide into clean and bad dataset
    train_set = np.concatenate([X_transformed, y_noisy], axis = 1)
    clean_idx = np.where(y_real == y_noisy)[0]
    all_idx = np.array(range(len(X)))
    bad_idx = np.setdiff1d(all_idx, clean_idx)
    train_clean_set = train_set[clean_idx, :]
    train_bad_set = train_set[bad_idx, :]

    # build train dataset, gt, noisy label and soft label
    train_clean_bad_set = np.concatenate((train_clean_set, train_bad_set), axis = 0)
    ground_truth = np.concatenate((y_real[clean_idx], y_real[bad_idx]), axis = 0)
    noisy_label = np.concatenate((y_noisy[clean_idx], y_noisy[bad_idx]), axis = 0)
    soft_label = np.concatenate((soft_label[clean_idx], soft_label[bad_idx]), axis = 0)

    # create dataloader
    my_dataset_tmp = MyDataSet(train_clean_bad_set)
    train_loader = DataLoader(my_dataset_tmp, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=None, pin_memory=False)
    eval_loader = DataLoader(my_dataset_tmp, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=None, pin_memory=False)

    return train_loader, eval_loader, train_clean_bad_set, train_clean_set, train_bad_set, ground_truth, noisy_label, soft_label
