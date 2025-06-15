# %% IMPORTS #
from NNfunctions import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
from torchvision.models import resnet18

from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Loss
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' for testing headless

import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% Structure for the NN: #
hyperVar = {
    'lr': 1e-4,
    'n_epochs': 20,
    'batch_size': 64,
    'device': device,
    'n_plotted': 10,
    'start': 10,
    'optimizer': Adam,
            }


class BasicBlock1D(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet18_1D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=3):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)

        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out).squeeze(-1)
        out = self.fc(out)
        return out

def resnet18_1d(num_classes=3):
    return ResNet18_1D(BasicBlock1D, [2, 2, 2, 2], num_classes)




# %% First, IMPORTING DATA #

trainSet = SignalDataset('model_creating_data/data.json', split="train")
testSet = SignalDataset('model_creating_data/data.json', split="test")

dataloader = DataLoader(trainSet, batch_size=hyperVar["batch_size"], shuffle=True)
test_loader = DataLoader(testSet, batch_size=hyperVar["batch_size"], shuffle=True)


# %% Defining training engine and Events

losses = []
eps = []
model = resnet18_1d()
# Here I Only intend to plot the losses graph so only need for basic event every 1/100 epochs
def supervised_train(model, optim=SGD, device="cpu", rate = 0.01):
    model.to(device)
    optimizer = optim(model.parameters(), lr=hyperVar["lr"])
    criterion = nn.MSELoss()
    losses = []
    eps = []

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()  ## This zeroes out the gradient stored in "model".
        X_batch, Y_batch = batch
        X, Y = X_batch.to(device), Y_batch.to(device)
        Y_pred = model(X)
        # Measure the loss/error, gonna be high at first
        loss = criterion(Y_pred, Y)  # predicted values vs the y_train
        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)

# Not really relevent for now and not used
def supervised_evaluator(model, device="cpu"):

    def _interference(engine, batch):
        model.eval()
        with torch.no_grad():
            X_batch, Y_batch = batch
            X, Y = X_batch.to(device), Y_batch.to(device)
            Y_pred = model(X)

            return Y_pred, Y

    engine = Engine(_interference)

    MeanRelativeError().attach(engine, 'rel_error_3')
    return engine

evaluator = supervised_evaluator(model, device=device)
trainer = supervised_train(model, optim=hyperVar['optimizer'], device=device, rate = hyperVar["lr"])

@trainer.on(Events.ITERATION_COMPLETED(every=1))
def log_training_loss(trainer):
    losses.append(trainer.state.output)
    eps.append(trainer.state.epoch)

# @trainer.on(Events.ITERATION_COMPLETED(every=int(n_epochs/50)))
# def early_stopping(trainer):
#     if trainer.state.output < 10**-8:
#         losses.append(trainer.state.output)
#         eps.append(trainer.state.epoch)
#         trainer.terminate()

ProgressBar().attach(trainer, output_transform=lambda x: {'loss': x})

# %% Training NN and recording events
# Will take some time
trainer.run(dataloader, max_epochs=hyperVar['n_epochs'])


# %%   PLOTTING THE FILTERING TO SEE IF IT WORKED   #
print("Python <<<<< legit everything else (except C)\n")
state = evaluator.run(test_loader)
rel_errors = state.metrics["rel_error_3"]
print(f"Relative Error (%): Left = {rel_errors[0]:.2f}, Center = {rel_errors[1]:.2f}, Right = {rel_errors[2]:.2f}")

# Seeing how the losses change:
plt.figure()
plt.semilogy(eps, losses)
plt.title("Loss vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")

# Plotting the dots:
n = hyperVar['n_plotted']
start = hyperVar['start']

fig, sigPlot = plt.subplots(nrows=1, ncols=n,  figsize=(4*n,6))
fig.text(0.6, 0.01, 'f [Hz]', ha='center', fontsize=12)
fig.text(0.01, 0.5, 'FFT', ha='center', rotation='vertical', fontsize=12)
import matplotlib.lines as mlines

# Create proxy artists
data_handle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='Data')
pred_handle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', label='Prediction')

# Add a figure-level legend
fig.legend(handles=[pred_handle, data_handle],
           loc='upper center', ncol=3, fontsize=10, frameon=False)
set = testSet
f = set.get_freqs()

X_params, Y_params = set.unscale()

for i_ in range(start,start+n):
    X_test,Y_test = set[i_]
    clean_sig = set.get_clean_sig(i_)
    peak_freqs = set.get_peak_freqs(i_)
    # peak_freqs = [peak_freqs[0].item(), 2000, peak_freqs[2].item()]

    if X_test.ndim == 2:
        X_test = X_test.unsqueeze(0)

    Y_pred = model(X_test.to(device))

    Y_pred_plot = Y_pred.cpu().detach().squeeze()
    Y_test_plot = Y_test.cpu().detach().squeeze()
    X_test_plot = X_test.cpu().detach().squeeze()

    X_test_plot = unscale_tensor(X_test_plot, X_params)
    Y_test_plot = unscale_tensor(Y_test_plot, Y_params)
    Y_pred_plot = unscale_tensor(Y_pred_plot, Y_params)

    Yn = Y_test_plot.numpy()

    i = i_%n

    sigPlot[i].scatter(peak_freqs, Yn,
                       edgecolors='black',
                       facecolors="none",
                       s=100,
                       zorder=5,
                       linewidths=2,
                       label='Real peak heights')

    sigPlot[i].scatter(peak_freqs, Y_pred_plot.numpy(),
                       edgecolors='blue',
                       facecolors="none",
                       s=100,
                       zorder=5,
                       linewidths=2,
                       label='Predicted peak heights')

    sigPlot[i].set_yscale('log')

    sigPlot[i].plot(f, X_test_plot.numpy(), "g*")
    sigPlot[i].semilogy(f, clean_sig, "r.-")

    # Add a title (number) to each column's top subplot
    sigPlot[i].set_title(f"n: {i}\n B0: {Y_test_plot[0]:.2e}")

    # xbor = [0, 4]
    # sigPlot[i].set_xlim(xbor[0],xbor[1])
#    sigPlot[i].set_xticks(np.linspace(xbor[0], xbor[1], 5))  # 11 ticks between 1950 and 2050
#    sigPlot[i].set_xticklabels([f"{xbor[0]:.0f}", "", "", "", f"{xbor[1]:.0f}"])

plt.tight_layout(rect=[0.03, 0.03, 1, 0.88])
plt.show()