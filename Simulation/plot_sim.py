# %% Imports
import sys
import os
import time
current_file = os.path.abspath(__file__)
research_path = os.path.dirname(os.path.dirname(current_file))
if research_path not in sys.path:
    sys.path.append(research_path)

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import torch
from torch.utils.data import DataLoader
from NNfunctions import *
from Simulation.numpy_ffts.fft_pink_noise import peak_heights_numpy
from matplotlib.colors import LogNorm

# Add timing debugging
start_time = time.time()
print(f"Script started at: {time.strftime('%H:%M:%S')}")

is_stft = True
unscale = False
real_b = False  # Use real B values for highlighting

# %% Plot configuration 
dataSet = 'train'  # Choose 'train', 'validate', or 'test'

# Choose which data set to show
start = 0
n = 3

# Create dataset and dataloader
if is_stft:
    file_name = 'Data/data_stft.h5'
else:
    file_name = 'Data/data.json'

# Create dataset using the SignalDataset class
print(f"Loading dataset from {file_name}...")
dataset_load_start = time.time()
dataset = SignalDataset(file_name, split=dataSet, same_scale=True)
dataset_load_time = time.time() - dataset_load_start
print(f"Dataset loaded in {dataset_load_time:.2f} seconds")
print(f"Dataset size: {len(dataset)} samples")

# Create a DataLoader
batch_size = n
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Get the first batch of data
data_fetch_start = time.time()
for batch_idx, (X_batch, Y_batch) in enumerate(dataloader):
    if batch_idx == 0:  # We only need the first batch
        X = X_batch
        Y = Y_batch
        break
data_fetch_time = time.time() - data_fetch_start
print(f"Data fetching completed in {data_fetch_time:.2f} seconds")
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# Get frequency and time information from the dataset
f = dataset.get_freqs().numpy()
if is_stft:
    freqs_stft = f
    time_bins = dataset.get_time_bins().numpy()
    print(f"STFT data dimensions - Frequencies: {freqs_stft.shape}, Time bins: {time_bins.shape}")

# Get F_B values for the first n samples
F_Bs = dataset.F_B[start:start+n].squeeze().numpy()

# Get original scaling parameters for potential unscaling
X_scale, Y_scale = dataset.unscale()

if unscale:
    X = unscale_tensor(X, X_scale)
    Y = unscale_tensor(Y, Y_scale)

# %% Plotting
print(f"Starting plotting at {time.strftime('%H:%M:%S')}")
plot_start_time = time.time()

if not is_stft:
    # Regular FFT plotting
    fig, sigPlot = plt.subplots(nrows=1, ncols=n, figsize=(4*n, 6))
    fig.text(0.6, 0.01, 'f [Hz]', ha='center', fontsize=12)
    fig.text(0.01, 0.5, 'FFT', ha='center', rotation='vertical', fontsize=12)

    # Create proxy artists
    data_handle = mlines.Line2D([], [], color='red', marker='.', linestyle='None', label='Data')
    pred_handle = mlines.Line2D([], [], color='blue', marker='*', linestyle='None', label='Prediction')

    for i in range(n):
        Xi, cleanSig = X[i], Y[i]

        F_B = F_Bs[i]
        
        # Get peak frequencies using dataset method
        peak_freqs = dataset.get_peak_freqs(i)
        highlight_x = peak_freqs
        highlight_y = peak_heights_numpy(cleanSig.numpy(), f_b=F_B, f_center=2000, dir=False)
        
        # Get clean signal for this index (if needed for special cases)
        clean_sig = dataset.get_clean_sig(i)

        if X_scale.get('log') and not unscale:
            sigPlot[i].plot(f, Xi, "g*")
            sigPlot[i].plot(f, cleanSig, "r.-")
        else:
            sigPlot[i].semilogy(f, Xi, "g*")
            sigPlot[i].semilogy(f, cleanSig, "r.-")

        sigPlot[i].scatter(highlight_x, highlight_y,
                          edgecolors='black',
                          facecolors="none",
                          s=100,
                          zorder=5,
                          linewidths=2,
                          label='Emphasized Points')

        # Add a title (number) to each column's top subplot
        sigPlot[i].set_title(f"n: {i}\n B0: {highlight_y[0]:.2e}")

        sigPlot[i].grid(True)
        # xbor = [1000, 3000]
        # sigPlot[i].set_xlim(xbor[0],xbor[1])
        # sigPlot[i].set_xticks([2000 - F_B, 2000 + F_B])  # 11 ticks between 1950 and 2050
        # sigPlot[i].set_xticklabels([f"{xbor[0]:.0f}", "", "", "", f"{xbor[1]:.0f}"])
else:
    # STFT plotting using imshow
    fig, sigPlot = plt.subplots(nrows=2, ncols=n, figsize=(4*n, 6))
    fig.text(0.6, 0.01, 'Time [s]', ha='center', fontsize=12)
    fig.text(0.01, 0.5, 'Frequency [Hz]', ha='center', rotation='vertical', fontsize=12)

    # Print time_bins and freqs's shapes for debugging
    print("Time bins shape:", np.shape(time_bins))
    print("Frequency bins shape:", np.shape(freqs_stft))

    for i in range(n):
        plot_iter_start = time.time()
        Xi, cleanSig = X[i], dataset.get_clean_sig(start + i)
        B0i = dataset.get_B0(start + i)

        if not unscale:
            cleanSig = scale_like_tensor(cleanSig, Y_scale)
            B0i = scale_like_tensor(B0i, Y_scale)

        # Convert tensors to numpy arrays if they aren't already
        if isinstance(Xi, torch.Tensor):
            Xi = Xi.numpy()
        if isinstance(cleanSig, torch.Tensor):
            cleanSig = cleanSig.numpy()

        F_B = F_Bs[i]
        
        # Add debugging output for large arrays
        if i == 0:  # Only print for the first iteration to avoid clutter
            print(f"Xi shape: {Xi.shape}, cleanSig shape: {cleanSig.shape}")

        # Plot STFT data using imshow
        if unscale:
            # Use LogNorm for scaled data
            im0 = sigPlot[0][i].imshow(Xi,
                           origin='lower',
                           aspect='auto',
                           cmap='viridis',
                           interpolation='none',
                           extent=[time_bins[0] if len(time_bins) > 1 else 0,
                                  time_bins[-1] if len(time_bins) > 1 else len(time_bins),
                                  freqs_stft[0] if len(freqs_stft) > 1 else 0,
                                  freqs_stft[-1] if len(freqs_stft) > 1 else len(freqs_stft)],
                           norm=LogNorm(vmin=max(Xi.min(), 1e-14), vmax=Xi.max()))

            im1 = sigPlot[1][i].imshow(cleanSig,
                           origin='lower',
                           aspect='auto',
                           cmap='viridis',
                           interpolation='none',
                           extent=[time_bins[0] if len(time_bins) > 1 else 0,
                                  time_bins[-1] if len(time_bins) > 1 else len(time_bins),
                                  freqs_stft[0] if len(freqs_stft) > 1 else 0,
                                  freqs_stft[-1] if len(freqs_stft) > 1 else len(freqs_stft)],
                           norm=LogNorm(vmin=max(cleanSig.min(), 1e-14), vmax=cleanSig.max()))
        else:
            # For log-magnitude data
            im0 = sigPlot[0][i].imshow(Xi,
                           origin='lower',
                           aspect='auto',
                           cmap='viridis',
                           interpolation='none',
                           extent=[time_bins[0] if len(time_bins) > 1 else 0,
                                  time_bins[-1] if len(time_bins) > 1 else len(time_bins),
                                  freqs_stft[0] if len(freqs_stft) > 1 else 0,
                                  freqs_stft[-1] if len(freqs_stft) > 1 else len(freqs_stft)])


            im1 = sigPlot[1][i].imshow(cleanSig,
                           origin='lower',
                           aspect='auto',
                           cmap='viridis',
                           interpolation='none',
                           extent=[time_bins[0] if len(time_bins) > 1 else 0,
                                  time_bins[-1] if len(time_bins) > 1 else len(time_bins),
                                  freqs_stft[0] if len(freqs_stft) > 1 else 0,
                                  freqs_stft[-1] if len(freqs_stft) > 1 else len(freqs_stft)])

        # Add colorbar to each subplot with its own values
        cbar = plt.colorbar(im0, ax=sigPlot[0][i])
        cbar.set_label('Magnitude' if unscale else 'Log Magnitude [dB]')
        cbary = plt.colorbar(im1, ax=sigPlot[1][i])
        cbary.set_label('Magnitude' if unscale else 'Log Magnitude [dB]')

        # Add title with F_B information
        sigPlot[0][i].set_title(f"Noisy | n: {i}\nF_B: {F_B} Hz\n B0: {B0i:.3e}")
        sigPlot[1][i].set_title(f"Clean | n: {i}\nF_B: {F_B} Hz\n B0: {B0i:.3e}")

        plot_iter_time = time.time() - plot_iter_start
        if i == 0 or i == n-1:  # Print timing for first and last iteration
            print(f"Plot iteration {i} completed in {plot_iter_time:.2f} seconds")

    # Create a new figure for plotting the FFT of a single time frame
    fig2, slicePlot = plt.subplots(nrows=1, ncols=n, figsize=(4*n, 4))
    fig2.suptitle('FFT of a Single Time Frame', fontsize=16)
    fig2.text(0.5, 0.01, 'Frequency [Hz]', ha='center', fontsize=12)
    fig2.text(0.01, 0.5, 'Magnitude', ha='center', rotation='vertical', fontsize=12)

    # Choose a time frame to plot (e.g., the middle one)
    time_idx = 2
    plot_time = time_bins[time_idx]

    for i in range(n):
        Xi = X[i].numpy() if isinstance(X[i], torch.Tensor) else X[i]
        Yi = Y[i].numpy() if isinstance(Y[i], torch.Tensor) else Y[i]
        cleanSig = dataset.get_clean_sig(start + i)
        B0i = dataset.get_B0(start + i)
        
        if not unscale:
            cleanSig = scale_like_tensor(cleanSig, Y_scale)
            B0i = scale_like_tensor(B0i, Y_scale)

        if isinstance(cleanSig, torch.Tensor):
            cleanSig = cleanSig.numpy()
            
        peak_freqs = dataset.get_peak_freqs(i)

        # Get the FFT for the chosen time frame
        Xi_slice = Xi[:, time_idx]
        Yi_slice = Yi
        clean_slice = cleanSig[:, time_idx]

        # Plot the FFT slices
        slicePlot[i].plot(freqs_stft, Xi_slice, 'g.-', label='Noisy')
        slicePlot[i].plot(freqs_stft, clean_slice, 'r.-', label='Clean')
        slicePlot[i].scatter(peak_freqs, Yi_slice,
                          edgecolors='black',
                          facecolors="none",
                          s=100,
                          zorder=5,
                          linewidths=2,
                          label='Emphasized Points')
        slicePlot[i].set_title(f"n: {i}, F_B: {F_B} Hz\n B0: {B0i:.3e}\nTime: {plot_time:.2f}s")
        slicePlot[i].grid(True)
        slicePlot[i].legend()
        if unscale:
            slicePlot[i].set_yscale('log')

plt.tight_layout(rect=[0.03, 0.03, 1, 0.88])

plt.show()

# %%
