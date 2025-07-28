# %% IMPORTS #
from NNfunctions import *
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Loss

import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' for testing headless
import matplotlib.pyplot as plt
import numpy as np




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%% PARAMS # 
hyperVar = {
    # Data parameters
    'batch_size': 16,
    'device': device,

    # Optimiser parameters
    'optimizer': Adam,
    'lr': 1e-5,
    'weight_decay': 1e-3,
    'beta1': 0.9, # ++ smoother training but slower response to changes
    'beta2': 0.999, # -- faster adaptation of learning rates but potentially less stability
    'amsgrad': True, 

    # Training procedure parameters
    'n_epochs': 10,
    'patience': 3,

    # Plotting parameters
    'n_plotted': 10,
    'start': 0,
    'unscaled_plot': True,
            }


# %% &&&&&& IMPORTING DATA &&&&&&
start_time = time.time()

trainSet = SignalDataset('Data/data_stft.h5', split="train", resnet=True)
validateSet = SignalDataset('Data/data_stft.h5', split="validate", resnet=True)
testSet = SignalDataset('Data/data_stft.h5', split="test", resnet=True)

dataloader = DataLoader(trainSet, batch_size=hyperVar["batch_size"], shuffle=True)
validate_loader = DataLoader(validateSet, batch_size=hyperVar["batch_size"], shuffle=False)
test_loader = DataLoader(testSet, batch_size=hyperVar["batch_size"], shuffle=False)

load_time = time.time() - start_time
print(f"Data loaded in {load_time:.2f} seconds")
print(f"Dataset sizes: Train={len(trainSet)}, Validation={len(validateSet)}, Test={len(testSet)}")


# %% Structure for the NN: #
class EfficientNetB0(nn.Module):
    def __init__(self, n_classes=3, pretrained=True, freeze_pretrained=False):
        """
        Initializes a EfficientNetB0 model adapted for single-channel input.
        Args:
            n_classes (int): The number of output values to predict.
            pretrained (bool): Whether to use pre-trained weights from ImageNet.
            freeze_pretrained (bool): If True, freezes all layers except the modified
                                     input and output layers.
        """
        super().__init__()
        # Load a pre-trained EfficientNetB0 model
        self.net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)

        # --- Freeze pre-trained layers if requested ---
        if pretrained and freeze_pretrained:
            for param in self.net.parameters():
                param.requires_grad = False

        # --- Adapt for 1-channel (grayscale) input ---
        original_conv = self.net.features[0][0]  # First conv layer in EfficientNet
        self.net.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        if pretrained:
            self.net.features[0][0].weight.data = original_conv.weight.data.sum(dim=1, keepdim=True)

        # --- Adapt the final layer for regression ---
        num_ftrs = self.net.classifier[1].in_features
        self.net.classifier[1] = nn.Linear(num_ftrs, n_classes)

        # --- Unfreeze the new layers so they can be trained ---
        if pretrained and freeze_pretrained:
            for param in self.net.features[0][0].parameters():
                param.requires_grad = True
            for param in self.net.classifier[1].parameters():
                param.requires_grad = True

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, 1, freq_bins, time_bins)
        Returns:
            torch.Tensor: The model's output.
        """
        return self.net(x)

# %% DELETING OLD PARAMETERS !!! 
model = EfficientNetB0(n_classes=3, pretrained=True, freeze_pretrained=False)
model = model.double()  # Convert model to float64

# %% ENGINE SETUP #

losses = []
val_losses = []
eps = []

criterion = nn.SmoothL1Loss()  # Mean Absolute Error (MAE)
optimizer = hyperVar['optimizer'](
    model.parameters(), 
    lr=hyperVar['lr'], 
    weight_decay=hyperVar['weight_decay'],
    amsgrad=hyperVar['amsgrad'],
    betas=(hyperVar['beta1'], hyperVar['beta2'])
)

def supervised_train(model, device="cpu"):
    model.to(device)
    optimizer.zero_grad() 

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


def supervised_evaluator(model, device="cpu"):

    def _interference(engine, batch):
        model.eval()
        with torch.no_grad():
            X_batch, Y_batch = batch
            X, Y = X_batch.to(device), Y_batch.to(device)
            Y_pred = model(X)

            return Y_pred, Y

    engine = Engine(_interference)

    # Attach loss metric
    Loss(criterion,
        output_transform=lambda x: (x[0], x[1])
    ).attach(engine, 'loss')

    return engine

trainer = supervised_train(model, device=device)

X_parms, Y_parms = testSet.unscale()
evaluator = supervised_evaluator(model, device=device)

handler = EarlyStopping(
    patience=hyperVar['patience'],  # Number of epochs to wait for improvement
    score_function=score_function,
    trainer=trainer
)
evaluator.add_event_handler(Events.COMPLETED, handler)

# Setup Model Checkpoint handler to save the best model
import os
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_handler = ModelCheckpoint(
    checkpoint_dir,
    filename_prefix="best_model",
    require_empty=False,
    score_function=score_function,  # Using the same score_function as early stopping
    score_name="val_loss",
    n_saved=1,  # Only save the best model
    global_step_transform=lambda engine, _: engine.state.epoch
)
evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler, {'model': model})

# Dictionary to store the best model parameters and corresponding score
best_model_info = {
    'params': None,
    'score': float('-inf'),  # Initialize with worst possible score (we want to maximize -loss)
    'epoch': 0
}

@trainer.on(Events.EPOCH_COMPLETED)
def run_validation_loss_track(engine):
    # Track losses for plotting
    eps.append(trainer.state.epoch)
    losses.append(trainer.state.output)

    # Run validation
    evaluator.run(validate_loader)
    val_loss = evaluator.state.metrics['loss']
    val_losses.append(val_loss)
    
    # Check if this is the best model so far
    current_score = -val_loss  # Negative because we want to maximize score (minimize loss)
    if current_score > best_model_info['score']:
        best_model_info['score'] = current_score
        best_model_info['params'] = {k: v.clone().detach() for k, v in model.state_dict().items()}
        best_model_info['epoch'] = trainer.state.epoch
        print(f"New best model found at epoch {trainer.state.epoch}!")
    
    print(f"Epoch {trainer.state.epoch} - Training Loss: {trainer.state.output:.4f}, "
          f"Validation Loss: {val_loss:.4f}, Best Epoch: {best_model_info['epoch']}")


ProgressBar().attach(trainer, output_transform=lambda x: {'loss': x})

# %% ********TRAINING*********
print("Starting training...")
# Ask user if they want to resume training from a checkpoint
resume = input("Do you want to resume training from a checkpoint? (y/n): ")
if resume.lower() == 'y':
    checkpoint_dir = "./checkpoints"
    files = os.listdir(checkpoint_dir)

    if files:
        checkpoint_path = os.path.join(checkpoint_dir, files[0])
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        print(f"Resuming training...")
    else:
        print("No checkpoint files found.")
else:
    print("Starting training from scratch...")
    
trainer.run(dataloader, max_epochs=hyperVar['n_epochs'])

# Restore the best model
if best_model_info['params'] is not None:
    print(f"Restoring best model from epoch {best_model_info['epoch']} with validation loss: {-best_model_info['score']:.4f}")
    model.load_state_dict(best_model_info['params'])
else:
    print("Warning: No best model was found during training!")


# %%   ------Evaluation of the model------
set = testSet
f = set.get_freqs()

X_params, Y_params = set.unscale()

def apply_unscaling(output):
    y_pred_scaled = unscale_tensor(output[0], Y_params)
    y_true_scaled = unscale_tensor(output[1], Y_params)
    return y_pred_scaled, y_true_scaled
transform_func = lambda x: apply_unscaling(x)

MeanRelativeError(
    output_transform=transform_func, device=device
).attach(evaluator, 'rel_error_3')

PeakComparison(
    output_transform=transform_func, device=device
).attach(evaluator, 'peak_comparison')


print("Python <<<<< legit everything else (except C)\n")
state = evaluator.run(test_loader)
rel_errors = state.metrics["rel_error_3"]
print(f"Relative Error (%) Y_pred / Y_true: Left = {rel_errors[0]:.2f}, Center = {rel_errors[1]:.2f}, Right = {rel_errors[2]:.2f}")

# %% -------Plotting evaluation--------

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

data_handle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='Data')
pred_handle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', label='Prediction')

# Add a figure-level legend
fig.legend(handles=[pred_handle, data_handle],
           loc='upper center', ncol=3, fontsize=10, frameon=False)

# Plotting the signals with the peaks
for i_ in range(start,start+n):
    X_test,Y_test = set[i_]
    clean_sig = set.get_clean_sig(i_)
    peak_freqs = set.get_peak_freqs(i_)

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

# Predicted vs True plots
Y_true_unscaled, Y_pred_unscaled, diffs = state.metrics["peak_comparison"]
Y_true_unscaled = Y_true_unscaled.cpu().numpy()
Y_pred_unscaled = Y_pred_unscaled.cpu().numpy()
diffs = diffs.cpu().numpy() * 1e12  # Convert to pT for better readability

# For MDE plot, we need intensities and diffs
intensities = Y_true_unscaled[:, 0]

# Plotting MDEPerIntensity
plt.figure(figsize=(10, 6))
plt.grid(True)
# plt.yscale('symlog', linthresh=1e-2)  # Log scale for better visibility of differences

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


peak_names = ['Left', 'Center', 'Right']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Predicted vs. True Peak Heights', fontsize=16)

for i, (ax, name) in enumerate(zip(axes, peak_names)):
    true_vals = Y_true_unscaled[:, i] * 1e12 # Convert to pT
    pred_vals = Y_pred_unscaled[:, i] * 1e12

    # Scatter plot
    ax.scatter(true_vals, pred_vals, alpha=0.5, label='Predictions')

    # Linear fit
    coeffs = np.polyfit(true_vals, pred_vals, 1)
    poly = np.poly1d(coeffs)
    fit_line = poly(true_vals)

    r_squared = np.corrcoef(true_vals, pred_vals)[0, 1] ** 2

    ax.plot(true_vals, fit_line, 'r-', label=f'Linear Fit (y={coeffs[0]:.2f}x + {coeffs[1]:.2e}, R²={r_squared:.2f})')

    ax.set_title(f'{name} Peak')
    ax.set_xlabel('True Height [pT]')
    ax.set_ylabel('Predicted Height [pT]')
    ax.legend()
    ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
# %%
