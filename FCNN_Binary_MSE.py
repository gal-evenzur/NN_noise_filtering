# %% IMPORTS #
from NNfunctions import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam

from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Loss

import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' for testing headless
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Parameters
hyperVar = {
    # Data parameters
    'batch_size': 32,
    'device': device,

    # Training parameters
    'optimizer': Adam,
    'lr': 1e-5,
    'n_epochs': 5,
    'patience': 5,

    # Plotting parameters
    'n_plotted': 10,
    'start': 0,
    'unscaled_plot': True,
            }

# %% First, IMPORTING DATA #

trainSet = SignalDataset('model_creating_data/data.json', split="train")
validateSet = SignalDataset('model_creating_data/data.json', split="validate")
testSet = SignalDataset('model_creating_data/data.json', split="test")

dataloader = DataLoader(trainSet, batch_size=hyperVar["batch_size"], shuffle=True)
validate_loader = DataLoader(validateSet, batch_size=hyperVar["batch_size"], shuffle=False)
test_loader = DataLoader(testSet, batch_size=hyperVar["batch_size"], shuffle=False)


# %% Structure for the NN: #

class PeakHeightRegressor(nn.Module):
    def __init__(self, f_signal=5001, h1=4096, h2=4096, h3=256, h4=128, h5=64, output=3):
        super().__init__()

        self.linear_stack = nn.Sequential(
            nn.Linear(f_signal, h1),
            nn.Linear(h1, h2),
            nn.Linear(h2, output),

        )


    def forward(self, x):
      logits = self.linear_stack(x)
      return logits



# %% Defining training engine and Events

losses = []
val_losses = []
eps = []
model = PeakHeightRegressor()
criterion = nn.MSELoss()
# Here I Only intend to plot the losses graph so only need for basic event every 1/100 epochs
def supervised_train(model, rate, optim, device="cpu"):
    model.to(device)
    optimizer = optim(model.parameters(), lr = rate)
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

    # Attach loss metric
    Loss(criterion,
        output_transform=lambda x: (x[0], x[1])
    ).attach(engine, 'loss')

    # Attach Mean Relative Error metric
    MeanRelativeError(
        output_transform=transform_func, device=device
    ).attach(engine, 'rel_error_3')

    # Attach MDEPerIntensity metric
    MDEPerIntensity(
        output_transform=transform_func, device=device
    ).attach(engine, 'mde_per_intensity')

    return engine

trainer = supervised_train(model, optim=hyperVar['optimizer'], device=device, rate = hyperVar["lr"])

X_parms, Y_parms = testSet.unscale()
evaluator = supervised_evaluator(model, device=device, scaling_params=Y_parms)

handler = EarlyStopping(
    patience=hyperVar['patience'],  # Number of epochs to wait for improvement
    score_function=score_function,
    trainer=trainer
)
evaluator.add_event_handler(Events.COMPLETED, handler)

@trainer.on(Events.EPOCH_COMPLETED)
def run_validation_loss_track(engine):
    # Track losses for plotting
    eps.append(trainer.state.epoch)
    losses.append(trainer.state.output)

    # Run validation
    evaluator.run(validate_loader)
    val_losses.append(evaluator.state.metrics['loss'])



ProgressBar().attach(trainer, output_transform=lambda x: {'loss': x})

# %% Training NN and recording events
# Will take some time
trainer.run(dataloader, max_epochs=hyperVar['n_epochs'])


# %%   PLOTTING THE FILTERING TO SEE IF IT WORKED   #
print("Python <<<<< legit everything else (except C)\n")
state = evaluator.run(test_loader)
rel_errors = state.metrics["rel_error_3"]
print(f"Relative Error (%) Y_pred / Y_true: Left = {rel_errors[0]:.2f}, Center = {rel_errors[1]:.2f}, Right = {rel_errors[2]:.2f}")

# Plotting MDEPerIntensity
intensities, diffs = state.metrics["mde_per_intensity"]  # intensities: [N,], diffs: [N,3]
# Convert to numpy if not already
intensities = intensities.cpu().numpy()
diffs = diffs.cpu().numpy() * 10**12  # Convert to pT for better readability

plt.figure(figsize=(10, 6))
plt.grid(True)
plt.yscale('symlog', linthresh=1e-2)  # Log scale for better visibility of differences

# plt.plot(intensities, diffs[:, 1], 'b.', label='Center Peak')
plt.plot(intensities, diffs[:, 0], 'r.', label='Left Peak')
plt.plot(intensities, diffs[:, 2], 'g.',label='Right Peak')
# Connect left and right dots for each intensity
for i in range(len(intensities)):
    plt.plot([intensities[i], intensities[i]], [diffs[i, 0], diffs[i, 2]], 'k-', alpha=0.5)
plt.title("Mean Deviation Error per Intensity")
plt.xlabel("Intensity (B0) [T]")
plt.ylabel("Mean Deviation Error [pT]")
plt.legend()

# Seeing how the losses change:
plt.figure()
plt.semilogy(eps, losses, label='Training Loss')
plt.semilogy(eps, val_losses, '--g', label='Validation Loss')
plt.title("Loss vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

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

    Y_pred = model(X_test.to(device))

    X_test_plot = X_test
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