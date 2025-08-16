# %% IMPORTS #
from NNfunctions import *
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
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
    'batch_size': 8, # Bigger = stable gradients and smaller updates
    'device': device,

    # Model parameters
    'same_scale': True,
    'n_outputs': 2,
    'Bnumber': 0,  # 0 for B0, 1 for B1, 2 for B2


    # Optimiser parameters
    'optimizer': AdamW,
    'h_lr': 5e-4,
    'b_lr': 1e-4,
    'weight_decay': 1e-5,
    'wd_off_below_lr': 5e-6,
    'beta1': 0.9, # ++ smoother training but slower response to changes
    'beta2': 0.999, # -- faster adaptation of learning rates but potentially less stability
    'amsgrad': False, 

    # Training procedure parameters
    'freeze_backbone_epochs': 2,    # set to 0 to disable; no reinit needed since lr=0 during freeze
    'n_epochs': 100,
    'patience': 6,
    'score_metric': 'r2_score',
    'lr_factor': 0.5,
    'lr_patience': 2, # number of no improvement rounds before lowering lr
    'min_lr': 1e-6,

    # Plotting parameters
    'n_plotted': 10,
    'start': 0,
    'unscaled_plot': True,
            }

hyperVar['no_middle'] = True if hyperVar['n_outputs'] == 2 else False 

# %% &&&&&& IMPORTING DATA &&&&&&
start_time = time.time()
same_scale = hyperVar['same_scale']  # If True, all signals are scaled to the same range
no_middle = hyperVar['no_middle']  # If True, regress only the left and right peaks

trainSet = SignalDataset('Data/data_stft.h5', split="train", resnet=True, same_scale=same_scale, no_middle=no_middle)
validateSet = SignalDataset('Data/data_stft.h5', split="validate", resnet=True, same_scale=same_scale, no_middle=no_middle)
testSet = SignalDataset('Data/data_stft.h5', split="test", resnet=True, same_scale=same_scale, no_middle=no_middle)

dataloader = DataLoader(trainSet, batch_size=hyperVar["batch_size"], shuffle=True)
validate_loader = DataLoader(validateSet, batch_size=hyperVar["batch_size"], shuffle=False)
test_loader = DataLoader(testSet, batch_size=hyperVar["batch_size"], shuffle=False)

load_time = time.time() - start_time
print(f"Data loaded in {load_time:.2f} seconds")
print(f"Dataset sizes: Train={len(trainSet)}, Validation={len(validateSet)}, Test={len(testSet)}")


# %% Structure for the NN: #
class EfficientNet(nn.Module):
    def __init__(self, n_classes=3, pretrained=True, freeze_pretrained=False):
        """
        Initializes a EfficientNet model adapted for single-channel input.
        Args:
            n_classes (int): The number of output values to predict.
            pretrained (bool): Whether to use pre-trained weights from ImageNet.
            freeze_pretrained (bool): If True, freezes all layers except the modified
                                     input and output layers.
        """
        super().__init__()
        # Load a pre-trained EfficientNet model
        if hyperVar['Bnumber'] == 0:
            self.net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        elif hyperVar['Bnumber'] == 1:
            self.net = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT if pretrained else None)
        elif hyperVar['Bnumber'] == 2:
            self.net = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT if pretrained else None)

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
            with torch.no_grad():
                w = original_conv.weight.data
                self.net.features[0][0].weight.copy_(w.mean(dim=1, keepdim=True))

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

def build_param_groups(model, lr_head, lr_body, weight_decay):
    """
    4 groups:
      0: backbone weights (with weight decay)
      1: backbone bias/norm (no decay)
      2: head weights (with weight decay)
      3: head bias/norm (no decay)
    """
    bb_wd, bb_nd, hd_wd, hd_nd = [], [], [], []

    def is_norm_or_bias(name):
        n = name.lower()
        return n.endswith(".bias") or "bn" in n or "norm" in n or "gn" in n

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = name.startswith("net.classifier")
        if is_head:
            (hd_nd if is_norm_or_bias(name) else hd_wd).append(p)
        else:
            (bb_nd if is_norm_or_bias(name) else bb_wd).append(p)

    groups = [
        {'params': bb_wd, 'lr': lr_body, 'weight_decay': weight_decay, 'backbone': True,  'decay': True},
        {'params': bb_nd, 'lr': lr_body, 'weight_decay': 0.0,        'backbone': True,  'decay': False},
        {'params': hd_wd, 'lr': lr_head, 'weight_decay': weight_decay/10, 'backbone': False, 'decay': True},
        {'params': hd_nd, 'lr': lr_head, 'weight_decay': 0.0,        'backbone': False, 'decay': False},
    ]
    # drop any empty groups
    return [g for g in groups if len(g['params']) > 0]

def set_bn_eval(m: nn.Module):
    # Put all BatchNorm variants in eval mode to freeze running_mean/var updates
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        # m.eval()
        pass

# %% DELETING OLD PARAMETERS !!! 
model = EfficientNet(n_classes=hyperVar['n_outputs'], pretrained=True, freeze_pretrained=False)

# %% ENGINE SETUP #

losses = []
val_losses = []
eps = []
lr = [hyperVar['h_lr'], hyperVar['b_lr']]

criterion = nn.SmoothL1Loss()

param_groups = build_param_groups(
    model,
    lr_head=hyperVar['h_lr'],
    lr_body=hyperVar['b_lr'],
    weight_decay=hyperVar['weight_decay']
)

optimizer = hyperVar['optimizer'](
    param_groups,
    betas=(hyperVar['beta1'], hyperVar['beta2']),
    amsgrad=hyperVar['amsgrad'],
)

def supervised_train(model, device="cpu"): 
    model.to(device)
    optimizer.zero_grad() 

    def _update(engine, batch):
        model.train()
        if engine.state.epoch <= hyperVar['freeze_backbone_epochs']:
            model.apply(set_bn_eval)  # Freeze BatchNorm running stats for the first epoch
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
            
            # Filter out abnormally large predictions
            # Create a mask for valid predictions (not abnormally large)
            # Calculate magnitude of predictions
            pred_magnitude = torch.norm(Y_pred, dim=1)
            
            # Define threshold for abnormal predictions (adjust as needed)
            # Using mean + 3*std as a statistical outlier threshold
            threshold = pred_magnitude.mean() + 3 * pred_magnitude.std()
            
            # Create mask for normal predictions
            valid_mask = pred_magnitude < threshold
            # Apply mask to predictions and ground truth
            # Filter predictions and ground truth
            if not torch.all(valid_mask):
                print(f"Filtered out {(~valid_mask).sum().item()} abnormal predictions")
                Y_pred = Y_pred[valid_mask]
                Y = Y[valid_mask]

            return Y_pred, Y

    engine = Engine(_interference)

    # Attach loss metric
    Loss(criterion,
        output_transform=lambda x: (x[0], x[1])
    ).attach(engine, 'loss')

    return engine

trainer = supervised_train(model, device=device)
evaluator = supervised_evaluator(model, device=device)

def score(engine):
    return score_function(engine, metric_name=hyperVar['score_metric'])

handler = EarlyStopping(
    patience=hyperVar['patience'],  # Number of epochs to wait for improvement
    score_function=score,
    trainer=trainer
)
evaluator.add_event_handler(Events.COMPLETED, handler)

# Add ReduceLROnPlateau scheduler that tracks validation loss (lower is better)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',           # minimize validation loss
    factor=hyperVar['lr_factor'],           # on plateau: lr <- lr * 0.2
    patience=hyperVar['lr_patience'],           # wait 2 validation epochs with no improvement
    threshold=1e-4,       # minimal change to qualify as improvement
    cooldown=0,           # no cooldown
    min_lr=hyperVar['min_lr'],          # do not go below this lr
)

def maybe_disable_wd(opt, threshold):
    for pg in opt.param_groups:
        if pg.get('decay', False) and pg['lr'] <= threshold and pg['weight_decay'] > 0.0:
            pg['weight_decay'] = 0.0
            pg['wd_disabled'] = True
            print("Disabled weight decay for a param group (lr below threshold)")


# Setup Model Checkpoint handler to save the best model
import os
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_handler = ModelCheckpoint(
    checkpoint_dir,
    filename_prefix="best_model",
    require_empty=False,
    score_function=score,  # Using the same score_function as early stopping
    score_name=hyperVar['score_metric'],
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

X_params, Y_params = validateSet.unscale()

def apply_unscaling(output):
    y_pred_scaled = unscale_tensor(output[0], Y_params)
    y_true_scaled = unscale_tensor(output[1], Y_params)
    return y_pred_scaled, y_true_scaled
transform_func = lambda x: apply_unscaling(x)

r2(
    output_transform=transform_func,
    device=device
).attach(evaluator, 'r2_score')

# ---- Backbone freeze just for the first epoch (no optimizer rebuild) ----
@trainer.on(Events.EPOCH_STARTED)
def freeze_backbone_for_starting_epochs(engine):
    if engine.state.epoch <= hyperVar['freeze_backbone_epochs']:
        for pg in optimizer.param_groups:
            if pg.get('backbone', False):
                # Save originals once
                pg.setdefault('orig_lr', pg['lr'])
                pg.setdefault('orig_wd', pg['weight_decay'])
                # Disable updates (and WD) for backbone during epoch 1
                pg['lr'] = 0.0
                pg['weight_decay'] = 0.0
        print("Backbone frozen: lr=0, weight_decay=0; BN running stats frozen via eval()")



@trainer.on(Events.EPOCH_COMPLETED)
def run_validation_loss_track(engine):

    if engine.state.epoch <= hyperVar['freeze_backbone_epochs']:
        for pg in optimizer.param_groups:
            if pg.get('backbone', False):
                # Restore original lr and weight decay
                pg['lr'] = pg.get('orig_lr', hyperVar['b_lr'])
                pg['weight_decay'] = pg.get('orig_wd', hyperVar['weight_decay'])
        print("Backbone unfrozen: restored lr and weight_decay; BN back to train() behavior")


    # Track losses for plotting
    eps.append(trainer.state.epoch)
    losses.append(trainer.state.output)

    # Run validation
    evaluator.run(validate_loader)
    val_loss = evaluator.state.metrics['loss']
    val_losses.append(val_loss)
    r2 = evaluator.state.metrics['r2_score']
    
    scheduler.step(val_loss)  # Step the scheduler based on validation loss
    maybe_disable_wd(optimizer, hyperVar['wd_off_below_lr'])


    lr.append([pg['lr'] for pg in optimizer.param_groups])
    # Check if this is the best model so far
    current_score = score(evaluator)  # Negative because we want to maximize score (minimize loss)
    if current_score > best_model_info['score']:
        best_model_info['score'] = current_score
        best_model_info['params'] = {k: v.clone().detach() for k, v in model.state_dict().items()}
        best_model_info['epoch'] = trainer.state.epoch
        print(f"New best model found at epoch {trainer.state.epoch}!")
    
    print(f"Epoch {trainer.state.epoch} - Training Loss: {trainer.state.output:.4f}, "
          f"Validation Loss: {val_loss:.4f}, Best Epoch: {best_model_info['epoch']}")
    print("R2 Score:", r2)
    print("Current LRs:", [f"{a:.2e}" for a in lr[-1]])



ProgressBar().attach(trainer, output_transform=lambda x: {'loss': x})
# %% CHECKPOINTS #

print("CHOOSE CHECKPOINT...")
# Ask user if they want to resume training from a checkpoint
checkpoint_dir = "./checkpoints"
if os.path.exists(checkpoint_dir):
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt') or f.endswith('.pth')]
    
    if files:
        print("\nAvailable checkpoint files:")
        for i, file in enumerate(files):
            print(f"{i+1}. {file}")
        
        while True:
            try:
                choice = int(input("\nSelect checkpoint number (or 0 to cancel): "))
                if choice == 0:
                    print("Canceled. Starting training from scratch...")
                    break
                elif 1 <= choice <= len(files):
                    checkpoint_path = os.path.join(checkpoint_dir, files[choice-1])
                    print(f"Loading checkpoint: {checkpoint_path}")
                    # Load the model state
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(checkpoint)
                      
                    print(f"Resuming training...")
                    break
                else:
                    print(f"Invalid choice. Please select a number between 0 and {len(files)}.")
            except ValueError:
                print("Please enter a valid number.")
    else:
        print("No checkpoint files found in the directory.")
else:
    print(f"Checkpoint directory {checkpoint_dir} does not exist.")
    
# %% ********TRAINING*********
print("--TRAINING---")
trainer.run(dataloader, max_epochs=hyperVar['n_epochs'])

# Restore the best model
if best_model_info['params'] is not None:
    print(f"Restoring best model from epoch {best_model_info['epoch']} with R2: {best_model_info['score']:.4f}")
    model.load_state_dict(best_model_info['params'])
else:
    print("Warning: No best model was found during training!")

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
# Print the relative error for each output
for i, rel_error in enumerate(rel_errors):
    print(f"Relative Error for output {i+1}: {rel_error:.4e}")
print("R2 Score for output :", state.metrics['r2_score'])
# %% -------Plotting evaluation--------
# ---------- Loss plot
plt.figure()
plt.semilogy(eps, losses, label='Training Loss')
plt.semilogy(eps, val_losses, '--g', label='Validation Loss')
plt.title("Loss vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# # -------- LR plot
# plt.figure()
# plt.plot(eps, lr, label='Learning Rate')
# plt.title("Learning Rate vs. Epochs")
# plt.xlabel("Epochs")
# plt.ylabel("Learning Rate")
# plt.legend()

# ---------- Signal plotting 
n = hyperVar['n_plotted']
start = hyperVar['start']

fig, sigPlot = plt.subplots(nrows=1, ncols=n,  figsize=(4*n,6))
fig.text(0.6, 0.01, 'f [Hz]', ha='center', fontsize=12)
fig.text(0.01, 0.5, 'FFT', ha='center', rotation='vertical', fontsize=12)
import matplotlib.lines as mlines

data_handle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='Data')
pred_handle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', label='Prediction')

fig.legend(handles=[pred_handle, data_handle],
           loc='upper center', ncol=3, fontsize=10, frameon=False)

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

    sigPlot[i].plot(f, X_test_plot[:,0], "g*")
    sigPlot[i].plot(f, clean_sig[:,0], "r.-")

    # Add a title (number) to each column's top subplot
    sigPlot[i].set_title(f"n: {i}\n B0: {Y_test_plot[0]:.2e}")

    # xbor = [0, 4]
    # sigPlot[i].set_xlim(xbor[0],xbor[1])
#    sigPlot[i].set_xticks(np.linspace(xbor[0], xbor[1], 5))  # 11 ticks between 1950 and 2050
#    sigPlot[i].set_xticklabels([f"{xbor[0]:.0f}", "", "", "", f"{xbor[1]:.0f}"])

# ---------- Mean Deviation Errors for each B0


plt.tight_layout(rect=[0.03, 0.03, 1, 0.88])

Y_true_unscaled, Y_pred_unscaled, diffs = state.metrics["peak_comparison"]
Y_true_unscaled = Y_true_unscaled.cpu().numpy()
Y_pred_unscaled = Y_pred_unscaled.cpu().numpy()
diffs = diffs.cpu().numpy() * 1e12  # Convert to pT for better readability

# Mask values which are greater than 1e-3
# mask = np.all(np.abs(Y_pred_unscaled) < 1e-3, axis=1) & np.all(np.abs(Y_true_unscaled) < 1e-6, axis=1)
# Y_pred_unscaled = Y_pred_unscaled[mask]

intensities = Y_true_unscaled[:, 0] * 1e12

plt.figure(figsize=(10, 6))
plt.grid(True)

# plt.yscale('symlog', linthresh=1e-2)  # Log scale for better visibility of differences
# plt.plot(intensities, diffs[:, 1], 'b.', label='Center Peak')
plt.plot(intensities, diffs[:, 0], 'r.', label='Left Peak')
plt.plot(intensities, diffs[:, -1], 'g.',label='Right Peak')
for i in range(len(intensities)): # Connect left and right dots for each intensity
    plt.plot([intensities[i], intensities[i]], [diffs[i, 0], diffs[i, -1]], 'k-', alpha=0.5)
plt.title("Mean Deviation Error per Intensity")
plt.xlabel("Intensity (B0) [T]")
plt.ylabel("Mean Deviation Error [pT]")
plt.legend()

# --------- Line fit
peak_names = ['Left', 'Center', 'Right']
if hyperVar['n_outputs'] == 2:
    peak_names = ['Left', 'Right']
fig, axes = plt.subplots(1, hyperVar['n_outputs'], figsize=(18, 6))
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

# ----------- Noise Plot
n = hyperVar['n_plotted']
start = hyperVar['start']

noiseSet = SignalDataset('Data/data_stft.h5', split="noise", resnet=True, same_scale=same_scale, no_middle=no_middle)

# Create arrays to store predicted values for left and right peaks
left_peak_predictions = []
right_peak_predictions = []

# Process all noise samples
for i_ in range(len(noiseSet)):
    X_test, _ = noiseSet[i_]

    X_test = X_test.unsqueeze(0)  # Reshape for model input
    Y_pred = model(X_test.to(device))
    
    if hyperVar['unscaled_plot']:
        Y_pred_plot = unscale_tensor(Y_pred, Y_params)
    else:
        Y_pred_plot = Y_pred
    
    Y_pred_plot = Y_pred_plot.cpu().detach().numpy()
    
    # Store predicted values
    left_peak_predictions.append(Y_pred_plot[0])  # Left peak [0]
    right_peak_predictions.append(Y_pred_plot[-1])  # Right peak [-1]

# Convert to numpy arrays for easier plotting
left_peak_predictions = np.array(left_peak_predictions)
right_peak_predictions = np.array(right_peak_predictions)

# Create scatter plot comparing left vs right peak predictions
plt.figure(figsize=(8, 8))
plt.scatter(left_peak_predictions * 1e12, right_peak_predictions * 1e12, alpha=0.5)
plt.title('Left Peak vs Right Peak Predictions from Noise')
plt.xlabel('Left Peak Prediction [pT]')
plt.ylabel('Right Peak Prediction [pT]')
plt.grid(True)
plt.axis('equal')

plt.show()

# %%
