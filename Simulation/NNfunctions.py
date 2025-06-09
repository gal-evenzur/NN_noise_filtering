import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
import json
import numpy as np
from model_creating_data.fft_pink_noise import peak_heights

def scale_tensor(dat_raw, log=False, norm=False, minmax=False, tensor=False):
    sc_par = {}
    # print(f"data is {dat_raw.shape} - {min.shape} / ({max.shape} - {min.shape} ")
    if tensor:
        lib = torch
    else:
        lib = np

    if log:
        dat_raw = lib.log10(dat_raw)
        sc_par['log'] = True

    if norm:
        mean = lib.mean(dat_raw)
        std = lib.std(dat_raw)
        dat_raw = (dat_raw - mean)/std
        sc_par['norm'] = [mean, std]


    if minmax:
        min = lib.min(dat_raw)
        max = lib.max(dat_raw)
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
        self.clean = clean_raw
        self.F_B = dat[split]["F_B"]

        sig_raw = torch.tensor(sig_raw, dtype=torch.float32)
        clean_raw = torch.tensor(clean_raw, dtype=torch.float32)
        self.F_B = torch.tensor(self.F_B, dtype=torch.int32).unsqueeze(1)

        amps = peak_heights(clean_raw, f_b=self.F_B, f_center=2000, dir=False)

        self.X, self.X_scale = transfrom(sig_raw, log=True, norm=True, tensor=True)
        self.Y, self.Y_scale = transfrom(amps, log=True, tensor=True)

        self.f = torch.FloatTensor(dat["f"])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_freqs(self):
        return self.f

    def get_peak_freqs(self, idx):
        return [2000 - self.F_B[idx].item(), 2000, 2000+self.F_B[idx].item()]

    def unscale(self):
        return self.X_scale, self.Y_scale

    def get_clean_sig(self, idx):
        return self.clean[idx]

