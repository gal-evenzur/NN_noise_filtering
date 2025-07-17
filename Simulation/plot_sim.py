# %% Imports
import sys
research_path = r"C:\Users\galev\Documents\Uni\Y3S2\Research"  # or use a dynamic path
if research_path not in sys.path:
    sys.path.append(research_path)


import matplotlib.pyplot as plt
# import colorednoise
import json
from NNfunctions import scale_tensor, unscale_tensor
from Simulation.numpy_ffts.fft_pink_noise import peak_heights_numpy
from matplotlib.colors import LogNorm

import numpy as np

is_stft = True
unscale= False
real_b = False  # Use real B values for highlighting

# %% Plot 10 of fft signal + fft clear signal #
file_name = 'Data/data.json' if not is_stft else 'Data/data_stft.json'
with open(file_name) as json_file:
    data = json.load(json_file)
dataSet = 'train'  # Choose 'train', 'validate', or 'test'

X = data[dataSet]['f_signals']
Y = data[dataSet]['f_cSignal']
f = data['f']
F_Bs = data[dataSet]["F_B"]

scale_train = {'log':True, 'norm':True, 'minmax':False}
scale_test = {'log':True, 'norm':False, 'minmax':False}
X, Xpar = scale_tensor(X, **scale_train)
Y, Ypar = scale_tensor(Y, **scale_test)

print("Successfully loaded data from", file_name)

# Choose which data set to show
start = 5
n = 3

if not is_stft:
    B0s = data[dataSet]["B_strength"]

    # Regular FFT plotting
    fig, sigPlot = plt.subplots(nrows=1, ncols=n, figsize=(4*n, 6))
    fig.text(0.6, 0.01, 'f [Hz]', ha='center', fontsize=12)
    fig.text(0.01, 0.5, 'FFT', ha='center', rotation='vertical', fontsize=12)
    import matplotlib.lines as mlines

    # Create proxy artists
    data_handle = mlines.Line2D([], [], color='red', marker='.', linestyle='None', label='Data')
    pred_handle = mlines.Line2D([], [], color='blue', marker='*', linestyle='None', label='Prediction')

    for i_ in range(start, start+n):
        Xi, Yi = X[i_], Y[i_]
        if unscale:
            Xi = unscale_tensor(Xi, Xpar)
            Yi = unscale_tensor(Yi, Ypar)

        F_B = F_Bs[i_]
        highlight_x = [2000 - F_B, 2000, 2000 + F_B]
        highlight_y = peak_heights_numpy(Yi, f_b=F_B, f_center=2000, dir=False)
        if real_b:
            highlight_y = [B0s[i_], highlight_y[1], B0s[i_]]

        i = i_%n

        if scale_train['log'] and not unscale:
            sigPlot[i].plot(f, Xi, "g*")
            sigPlot[i].plot(f, Yi, "r.-")
        else:
            sigPlot[i].semilogy(f, Xi, "g*")
            sigPlot[i].semilogy(f, Yi, "r.-")

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

    # Extract time and frequency data if available
    time_bins = data['t']
    freqs_stft = data['f']

    # Print time_bins and freqs's shapes for debugging
    print("Time bins shape:", np.shape(time_bins))
    print("Frequency bins shape:", np.shape(freqs_stft))


    for i_ in range(start, start+n):
        Xi, Yi = X[i_], Y[i_]
        if unscale:
            Xi = unscale_tensor(Xi, Xpar)
            Yi = unscale_tensor(Yi, Ypar)

        F_B = F_Bs[i_]
        i = i_%n

        # Plot STFT data using imshow
        if unscale:
            # Use LogNorm for scaled data
            im = sigPlot[0][i].imshow(Xi,
                           origin='lower',
                           aspect='auto',
                           cmap='viridis',
                           interpolation='none',
                           extent=[time_bins[0] if len(time_bins) > 1 else 0,
                                  time_bins[-1] if len(time_bins) > 1 else len(time_bins),
                                  freqs_stft[0] if len(freqs_stft) > 1 else 0,
                                  freqs_stft[-1] if len(freqs_stft) > 1 else len(freqs_stft)],
                           norm=LogNorm(vmin=max(Xi.min(), 1e-14), vmax=Xi.max()))

            im = sigPlot[1][i].imshow(Yi,
                           origin='lower',
                           aspect='auto',
                           cmap='viridis',
                           interpolation='none',
                           extent=[time_bins[0] if len(time_bins) > 1 else 0,
                                  time_bins[-1] if len(time_bins) > 1 else len(time_bins),
                                  freqs_stft[0] if len(freqs_stft) > 1 else 0,
                                  freqs_stft[-1] if len(freqs_stft) > 1 else len(freqs_stft)],
                           norm=LogNorm(vmin=max(Yi.min(), 1e-14), vmax=Yi.max()))
        else:
            # For log-magnitude data
            im = sigPlot[0][i].imshow(Xi,
                           origin='lower',
                           aspect='auto',
                           cmap='viridis',
                           interpolation='none',
                           extent=[time_bins[0] if len(time_bins) > 1 else 0,
                                  time_bins[-1] if len(time_bins) > 1 else len(time_bins),
                                  freqs_stft[0] if len(freqs_stft) > 1 else 0,
                                  freqs_stft[-1] if len(freqs_stft) > 1 else len(freqs_stft)])


            im = sigPlot[1][i].imshow(Yi,
                           origin='lower',
                           aspect='auto',
                           cmap='viridis',
                           interpolation='none',
                          extent=[time_bins[0] if len(time_bins) > 1 else 0,
                                  time_bins[-1] if len(time_bins) > 1 else len(time_bins),
                                  freqs_stft[0] if len(freqs_stft) > 1 else 0,
                                  freqs_stft[-1] if len(freqs_stft) > 1 else len(freqs_stft)])

        # Add colorbar to each subplot
        cbar = plt.colorbar(im, ax=sigPlot[0][i])
        cbar.set_label('Magnitude' if unscale else 'Log Magnitude [dB]')
        cbary = plt.colorbar(im, ax=sigPlot[1][i])
        cbary.set_label('Magnitude' if unscale else 'Log Magnitude [dB]')

        # Add title with F_B information
        sigPlot[0][i].set_title(f"Noisy | n: {i}\nF_B: {F_B} Hz")
        sigPlot[1][i].set_title(f"Clean | n: {i}\nF_B: {F_B} Hz")

    # Create a new figure for plotting the FFT of a single time frame
    fig2, slicePlot = plt.subplots(nrows=1, ncols=n, figsize=(4*n, 4))
    fig2.suptitle('FFT of a Single Time Frame', fontsize=16)
    fig2.text(0.5, 0.01, 'Frequency [Hz]', ha='center', fontsize=12)
    fig2.text(0.01, 0.5, 'Magnitude', ha='center', rotation='vertical', fontsize=12)

    # Choose a time frame to plot (e.g., the middle one)
    time_idx = 2
    plot_time = time_bins[time_idx]

    for i_ in range(start, start+n):
        Xi, Yi = X[i_].squeeze(), Y[i_].squeeze()
        if unscale:
            Xi = unscale_tensor(Xi, Xpar)
            Yi = unscale_tensor(Yi, Ypar)

        F_B = F_Bs[i_]
        i = i_%n

        # Get the FFT for the chosen time frame
        Xi_slice = Xi[:, time_idx]
        Yi_slice = Yi[:, time_idx]

        # Plot the FFT slices
        slicePlot[i].plot(freqs_stft, Xi_slice, 'g.-', label='Noisy')
        slicePlot[i].plot(freqs_stft, Yi_slice, 'r.-', label='Clean')
        slicePlot[i].set_title(f"n: {i}, F_B: {F_B} Hz\nTime: {plot_time:.2f}s")
        slicePlot[i].grid(True)
        slicePlot[i].legend()
        if unscale:
            slicePlot[i].set_yscale('log')


plt.tight_layout(rect=[0.03, 0.03, 1, 0.88])

plt.show()
'''
# %%  PLOTTING NOISE   #

dt = 1e-4  # 10 kHz sample rate
fs = 1 / dt
n_power = 2.5e-5
pink_percentage = 0.5  # 50% pink, 50% white for the mix

# Generate noises
combined_noise, pink_noise, white_noise = make_noise(dt, n_power, pink_percentage)

# Welch Power Spectral Density
f_w, Pxx_w = welch(white_noise, fs, nperseg=1024)
f_p, Pxx_p = welch(pink_noise, fs, nperseg=1024)
f_c, Pxx_c = welch(combined_noise, fs, nperseg=1024)

# Plotting (with log-log for max nerd power)
fig, Noise = plt.subplots(nrows=3, figsize=(10, 6))

Noise[0].loglog(f_w, Pxx_w, label='White Noise')
Noise[1].loglog(f_p, Pxx_p, label='Pink Noise')
Noise[2].loglog(f_c, Pxx_c, label='Combined Noise (50/50)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectral Density [V²/Hz]')
plt.title('Welch Power Spectra: White vs Pink vs Combined')
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()

#   Add reference lines for noises  #

ref_p = Pxx_p[50] * (f_p[50] / f_p[1:])  # reference line ~1/f
Noise[1].loglog(f_p[1:], ref_p, 'k--', label='~1/f')


# %% Noise with slider

def plot_spectrum(ax, noise, fs):
    ax.clear()
    f, Pxx = welch(noise, fs, nperseg=1024)
    ax.loglog(f, Pxx, label="Combined Noise")

    # Flat line (white noise reference)
    flat_val = np.mean(Pxx)
    ax.loglog(f, [flat_val] * len(f), 'k--', label="Flat Fit (White)")

    # 1/f line (pink noise reference)
    f_nonzero = f[f > 0]
    A = Pxx[1] * f_nonzero[0]
    pink_line = A / f_nonzero
    ax.loglog(f_nonzero, pink_line, 'r--', label="1/f Fit (Pink)")

    ax.set_title("Power Spectrum (Welch)")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power Spectral Density")
    ax.grid(True, which='both', ls='--')
    ax.legend()

# ------------- GUI App -------------

def update_plot(_):
    pink_perc = pink_slider.get() / 100
    power_val = 10 ** (power_slider.get())  # Exponential scale
    noise, _, _ = make_noise(dt=1e-4, n_power=power_val, p_perc=pink_perc)
    plot_spectrum(ax, noise, fs=1/1e-4)
    canvas.draw()


root = tk.Tk()
root.title("Noise Mixer (White vs Pink)")

fig, ax = plt.subplots(figsize=(7, 5))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()


pink_slider = tk.Scale(root, from_=0, to=100, orient='horizontal', label="Pink Noise %", command=update_plot)
pink_slider.set(50)
pink_slider.pack()

power_slider = tk.Scale(root, from_=-6, to=-0, resolution=0.1, orient='horizontal', label="Log10 Noise Power", command=update_plot)
power_slider.set(-4.6)  # ~2.5e-5
power_slider.pack()

update_plot(None)

plt.show()
root.mainloop()
'''
