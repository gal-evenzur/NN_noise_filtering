import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
import json

def scale_tensor(dat_raw):
    dat_raw = torch.FloatTensor(dat_raw)
    min = dat_raw.amin(1)
    max = dat_raw.amax(1)
    # print(f"data is {dat_raw.shape} - {min.shape} / ({max.shape} - {min.shape} ")

    dat_norm = (dat_raw - min[:, None])/(max[:, None] - min[:, None])
    # std = dat_norm.std(1)
    # dat_norm = dat_norm/std[:, None]
    return dat_norm


class SignalDataset(Dataset):
    def __init__(self, json_path, split="train", transfrom=scale_tensor):

        with open(json_path, "r") as f:
            dat = json.load(f)
        x_raw = dat[split]["f_signals"]
        y_raw = dat[split]["f_cSignal"]

        self.X = transfrom(x_raw)
        self.Y = transfrom(y_raw)

        self.f = torch.FloatTensor(dat["f"])

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_freqs(self):
        return self.f


