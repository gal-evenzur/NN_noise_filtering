# %% IMPORTS #
from NNfunctions import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Loss
import matplotlib

from resnet1d_master.resnet1d import ResNet1D

matplotlib.use('TkAgg')  # or 'Agg' for testing headless

import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Parameters
hyperVar = {
    'lr': 1e-2,
    'n_epochs': 5,
    'batch_size': 32,
    'device': device,
    'n_plotted': 10,
    'start': 0,
    'optimizer': Adam,
    'unscaled_plot': True,
            }

# %% Structure for the NN: #

model = ResNet1D(
    in_channels=1,
    base_filters=128,
    kernel_size=16,
    stride=2,
    n_block=18,
    groups=1,
    n_classes=3,
    downsample_gap=6,
    increasefilter_gap=12,
    use_bn=True,
    use_do=True,
    verbose=False)
model.to(device)



# %% First, IMPORTING DATA #

trainSet = SignalDataset('model_creating_data/data.json', split="train", resnet=True)
testSet = SignalDataset('model_creating_data/data.json', split="test", resnet=True)

dataloader = DataLoader(trainSet, batch_size=hyperVar["batch_size"], shuffle=True)
test_loader = DataLoader(testSet, batch_size=hyperVar["batch_size"], shuffle=True)


# %% Defining training engine and Events

losses = []
eps = []


# Here I Only intend to plot the losses graph so only need for basic event every 1/100 epochs
def supervised_train(model, rate, optim, device="cpu"):
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


def supervised_evaluator(model, device="cpu", scaling_params=None):

    def _interference(engine, batch):
        model.eval()
        with torch.no_grad():
            X_batch, Y_batch = batch
            X, Y = X_batch.to(device), Y_batch.to(device)
            Y_pred = model(X)

            return Y_pred, Y

    engine = Engine(_interference)

    if scaling_params:
        def apply_unscaling(output):
            y_pred_scaled = unscale_tensor(output[0], scaling_params)
            y_true_scaled = unscale_tensor(output[1], scaling_params)
            return y_pred_scaled, y_true_scaled

        transform_func = apply_unscaling
    else:
        transform_func = lambda x: x

    MeanRelativeError(
        output_transform=transform_func, device=device
    ).attach(engine, 'rel_error_3')
    return engine

trainer = supervised_train(model, optim=hyperVar['optimizer'], device=device, rate = hyperVar["lr"])

X_parms, Y_parms = testSet.unscale()
evaluator = supervised_evaluator(model, device=device, scaling_params=Y_parms)

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
print(f"Relative Error (%) Y_pred / Y_true: Left = {rel_errors[0]:.2f}, Center = {rel_errors[1]:.2f}, Right = {rel_errors[2]:.2f}")


# Seeing how the losses change:
plt.figure()
plt.semilogy(eps, losses)
plt.title("Loss vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
print("Losses printed, now plotting the graphs\n")

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

    X_test = X_test.unsqueeze(0)  # Reshape to [1, channels, length] for ResNet
    Y_pred = model(X_test.to(device))

    X_test_plot = X_test.squeeze()
    Y_test_plot = Y_test
    Y_pred_plot = Y_pred

    if hyperVar['unscaled_plot']:
        X_test_plot = unscale_tensor(X_test_plot, X_params)
        Y_test_plot = unscale_tensor(Y_test_plot, Y_params)
        Y_pred_plot = unscale_tensor(Y_pred_plot, Y_params)

    Y_pred_plot = Y_pred_plot.cpu().detach().numpy()
    Y_test_plot = Y_test_plot.cpu().detach().numpy()
    X_test_plot = X_test_plot.cpu().detach().numpy()


    i = i_%n

    sigPlot[i].scatter(peak_freqs, Y_test_plot,
                       edgecolors='black',
                       facecolors="none",
                       s=100,
                       zorder=5,
                       linewidths=2,
                       label='Real peak heights')

    sigPlot[i].scatter(peak_freqs, Y_pred_plot,
                       edgecolors='blue',
                       facecolors="none",
                       s=100,
                       zorder=5,
                       linewidths=2,
                       label='Predicted peak heights')

    if hyperVar['unscaled_plot']:
        sigPlot[i].set_yscale('log')

    sigPlot[i].plot(f, X_test_plot, "g*")
    sigPlot[i].plot(f, clean_sig, "r.-")

    # Add a title (number) to each column's top subplot
    sigPlot[i].set_title(f"n: {i}\n B0: {Y_test_plot[0]:.2e}")

    # xbor = [0, 4]
    # sigPlot[i].set_xlim(xbor[0],xbor[1])
#    sigPlot[i].set_xticks(np.linspace(xbor[0], xbor[1], 5))  # 11 ticks between 1950 and 2050
#    sigPlot[i].set_xticklabels([f"{xbor[0]:.0f}", "", "", "", f"{xbor[1]:.0f}"])

plt.tight_layout(rect=[0.03, 0.03, 1, 0.88])
plt.show()