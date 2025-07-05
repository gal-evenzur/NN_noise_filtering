from fft_pink_noise import *
import json
from math import gcd
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
import numpy as np  # Add this if not already imported

# Parameters for the voltage signal
I0 = 50e-3  # The current amplitude in the sensor[A]
B0 = 4e-12  # The magnetic field on the sensor [T]
F_B = 15  # The magnetic field frequency [Hz]
noise_strength = 0.5e-10

# Sampling and time parameters


# Stft parameters
fs = 6000
total_cycles = 100
overlap_perc = 0.75
cycles_per_window = 5  # Number of cycles in one window

dt = 1 / fs
freqs = [2000 - F_B, 2000, 2000 + F_B]  # Frequencies of interest
# Compute GCD
freq_gcd = reduce(gcd, freqs)

# Period is 1 / GCD of all frequency components
Tperiod = 1 / freq_gcd

print(f"Estimated period of the voltage signal: {Tperiod:.10f} seconds")

end_time = Tperiod * total_cycles


result = Signal_Noise_FFts(I0, B0, F_B, noise_strength,
                          dt=dt,
                          start_time=0,
                          end_time=end_time, use_torch=False)

fSignal = result[3]
frequencies = result[5]
if not isinstance(fSignal, torch.Tensor):
    signal = torch.from_numpy(result[2])
else:
    signal = result[2]
Time = result[4]

# Use my_stft function instead of manual STFT calculation
magnitude, freqs_stft, time_bins = my_stft(I0, B0, F_B, noise_strength,
                              fs=fs,
                              total_cycles=total_cycles,
                              overlap=overlap_perc,
                              cycles_per_window=cycles_per_window,
                              only_center=True,
                              use_torch=False)


# Plot spectrogram (log-magnitude, 1900–2100 Hz)
plt.figure(figsize=(10, 4))
plt.imshow(magnitude,
           origin='lower',
           aspect='auto',
           cmap='viridis',
           interpolation='none',
           extent=[time_bins[0], time_bins[-1], freqs_stft[0], freqs_stft[-1]],
           norm=LogNorm(vmin=1e-14, vmax=magnitude.max().item())  # log color scale
           )

plt.colorbar(label='Log Magnitude [dB]')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title('STFT Log-Magnitude (1900–2100 Hz)')
plt.tight_layout()

# Create a subplot figure with both the full FFT and a single frame STFT
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Full FFT
axs[0].plot(frequencies, np.abs(fSignal))
axs[0].set_yscale('log')
axs[0].set_xlabel('Frequency [Hz]')
axs[0].set_ylabel('Magnitude')
axs[0].set_title('Full FFT of Signal')
axs[0].grid(True)

# Plot 2: Single frame from STFT
frame_idx = magnitude.shape[1] // 2
axs[1].plot(freqs_stft, magnitude[:, frame_idx])
axs[1].set_yscale('log')  # Use log scale for better visualization
axs[1].set_xlabel('Frequency [Hz]')
axs[1].set_ylabel('Magnitude')
axs[1].grid(True)
axs[1].set_ylabel('Log Magnitude [dB]')
plt.show()
