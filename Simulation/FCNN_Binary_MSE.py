# %% IMPORTS #
from NNfunctions import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar

import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' for testing headless

import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% Structure for the NN: #
hyperVar = {
    'lr': 0.1,
    'n_epochs': 400,
    'batch_size': 64,
    'device': device,
    'n_plotted': 10,
    'start': 0,
    'optimizer': Adam,
            }

class PeakHeightRegressor(nn.Module):
    def __init__(self, f_signal=5001, h1=512, h2=256, h3=128, h4=64, h5=32, output=3):
        super().__init__()

        self.linear_stack = nn.Sequential(
            nn.Linear(f_signal, h1),
            nn.Linear(h1, h2),
            nn.Linear(h2, h3),
            nn.Linear(h3, h4),
            nn.Linear(h4, h5),
            nn.Linear(h5, output),
        )


    def forward(self, x):
      logits = self.linear_stack(x)
      return logits




# %% First, IMPORTING DATA #

trainSet = SignalDataset('model_creating_data/data.json', split="train")
testSet = SignalDataset('model_creating_data/data.json', split="test")

dataloader = DataLoader(trainSet, batch_size=hyperVar["batch_size"], shuffle=True)

'''
# Example of the scaled versions of the signals: #
fig, sigPlot = plt.subplots(nrows=2, ncols=5)


x_batch, y_batch = next(iter(dataloader))
f = trainSet.get_freqs()

for i in range(5):

    sigPlot[0][i].semilogy(f, x_batch[i], ".")
    sigPlot[1][i].semilogy(f, y_batch[i], "r.")

    # sigPlot[0][i].set_xlim(1950,2050)
    # sigPlot[0][i].set_ylim(0.00001,10)
    # sigPlot[1][i].set_xlim(1950,2050)
    # sigPlot[1][i].set_ylim(0.00001,10)

plt.show()
'''

# %% Defining training engine and Events

losses = []
eps = []
model = PeakHeightRegressor()
# Here I Only intend to plot the losses graph so only need for basic event every 1/100 epochs
def supervised_train(model, optim=SGD, device="cpu", rate = 0.01):
    model.to(device)
    optimizer = optim(model.parameters(), lr = rate)
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

            return Y_pred, Y_batch
    engine = Engine(_interference)
    return engine


trainer = supervised_train(model, optim=hyperVar['optimizer'], device=device, rate = hyperVar["lr"])

@trainer.on(Events.ITERATION_COMPLETED(every=int(hyperVar['n_epochs'] / 100)))
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

# Seeing how the losses change:
plt.figure()
plt.semilogy(eps, losses)
plt.title("Loss vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")

# Plotting the dots:
n = hyperVar['n_plotted']
start = hyperVar['start']

fig, sigPlot = plt.subplots(nrows=2, ncols=n,  figsize=(4*n,6))
fig.text(0.6, 0.01, 'f [Hz]', ha='center', fontsize=12)
fig.text(0.01, 0.5, 'FFT', ha='center', rotation='vertical', fontsize=12)
import matplotlib.lines as mlines

# Create proxy artists
data_handle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='Data')
pred_handle = mlines.Line2D([], [], color='blue', marker='*', linestyle='None', label='Prediction')

# Add a figure-level legend
fig.legend(handles=[pred_handle, data_handle],
           loc='upper center', ncol=3, fontsize=10, frameon=False)

f = testSet.get_freqs()

X_params, Y_params = testSet.unscale()

for i_ in range(start,start+n):
    X_test,Y_test = testSet[i_]
    peak_freqs = testSet.get_peak_freqs(i_)

    Y_pred = model(X_test.to(device))

    Y_p_numpy = Y_pred.cpu().detach().numpy()
    Y_t_numpy = Y_test.cpu().detach().numpy()
    X_t_numpy = X_test.cpu().detach().numpy()

    X_test_plot = unscale_tensor(X_t_numpy, X_params)
    Y_test_plot = unscale_tensor(Y_t_numpy, Y_params)
    Y_pred_plot = unscale_tensor(Y_p_numpy, Y_params)

    i = i_%n

    sigPlot[0][i].scatter(peak_freqs, Y_test_plot,
                       edgecolors='black',
                       facecolors="none",
                       s=100,
                       zorder=5,
                       linewidths=2,
                       label='Real peak heights')

    sigPlot[0][i].scatter(peak_freqs, Y_pred_plot,
                       edgecolors='red',
                       facecolors="none",
                       s=100,
                       zorder=5,
                       linewidths=2,
                       label='Predicted peak heights')

    sigPlot[1][i].semilogy(f, X_t_numpy, ".-")

    # Add a title (number) to each column's top subplot
    sigPlot[0][i].set_title(f"n: {i}\n B0: {Y_test_plot[0]:.2e}")

    # xbor = [1000, 3000]
    # sigPlot[0][i].set_xlim(xbor[0],xbor[1])
#    sigPlot[0][i].set_xticks(np.linspace(xbor[0], xbor[1], 5))  # 11 ticks between 1950 and 2050
#    sigPlot[0][i].set_xticklabels([f"{xbor[0]:.0f}", "", "", "", f"{xbor[1]:.0f}"])

plt.tight_layout(rect=[0.03, 0.03, 1, 0.88])
plt.show()