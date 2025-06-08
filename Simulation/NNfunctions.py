import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
import json
import numpy as np

def scale_tensor(dat_raw, log=False, norm=False, minmax=False):
    sc_par = {}
    min = np.min(dat_raw)
    max = np.max(dat_raw)
    # print(f"data is {dat_raw.shape} - {min.shape} / ({max.shape} - {min.shape} ")

    if log:
        dat_raw = np.log10(dat_raw)
        sc_par['log'] = True

    if norm:
        mean = np.mean(dat_raw)
        std = np.std(dat_raw)
        dat_raw = (dat_raw - mean)/std
        sc_par['norm'] = [mean, std]


    if minmax:
        dat_raw = (dat_raw - min)/(max - min)
        sc_par['minmax'] = [min, max]

    return dat_raw, sc_par

def unscale_tensor(procss_data, params):
    if params.get('minmax'):
        mn = params.get('minmax')[0]
        mx = params.get('minmax')[1]
        procss_data = procss_data * (mx - mn) + mn

    if params.get('norm'):
        # First multiply by std, then add the mean
        procss_data = (procss_data * params.get('norm')[1]) + params.get('norm')[0]


    if params.get('log'):
        procss_data = 10**procss_data

    return procss_data


class SignalDataset(Dataset):
    def __init__(self, json_path, split="train", transfrom=scale_tensor):

        with open(json_path, "r") as f:
            dat = json.load(f)
        sig_raw = dat[split]["f_signals"]
        clean_raw = dat[split]["f_cSignal"]
        self.F_B = dat[split]["F_B"]

        self.X = transfrom(sig_raw)
        self.Y = transfrom(clean_raw)

        self.f = torch.FloatTensor(dat["f"])

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_freqs(self):
        return self.f

    def get_amps(self):
        return self.B


