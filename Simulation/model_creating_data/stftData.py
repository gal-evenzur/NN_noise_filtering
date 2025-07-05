from fft_pink_noise import *
import json
from math import gcd
from functools import reduce
import matplotlib.pyplot as plt
import torch
import numpy as np  # Add this if not already imported

# Parameters for the voltage signal
I0 = 50e-3  # The current amplitude in the sensor[A]
B0 = 8e-12  # The magnetic field on the sensor [T]
F_B = 15  # The magnetic field frequency [Hz]
noise_strength = 0.5e-11

# Sampling and time parameters

# Convert to integers: assume Hz are integers, so periods are rational
# Find LCM of the frequencies' denominators to get common period

def lcm(a, b):
    return abs(a * b) // gcd(a, b)

def lcm_multiple(numbers):
    return reduce(lcm, numbers)


fs = 6000
dt = 1 / fs
freqs = [2000 - F_B, 2000, 2000 + F_B]  # Frequencies of interest
# Compute GCD
freq_gcd = reduce(gcd, freqs)

# Period is 1 / GCD of all frequency components
Tperiod = 1 / freq_gcd

print(f"Estimated period of the voltage signal: {Tperiod:.10f} seconds")

N_windows = 30
end_time = Tperiod * N_windows


result = Signal_Noise_FFts(I0, B0, F_B, noise_strength,
                          dt=dt,
                          start_time=0,
                          end_time=end_time,
                          is_padding=False,
                          is_window=False,
                          only_noise=False)

fSignal = result[3]
frequencies = result[5]
signal = torch.from_numpy(result[2])
Time = result[4]

# Compute STFT
cycles_per_window = 4  # Number of cycles in one window
n_fft = int(fs * Tperiod) * cycles_per_window
hop_length = n_fft // 5
win_length = n_fft
window = torch.hann_window(win_length)

stft_result = torch.stft(signal, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length,
                         window=None,
                         return_complex=True,
                         center=False,)

# Convert to magnitude and then to log-magnitude
magnitude = stft_result.abs().numpy()
log_magnitude = 20 * np.log10(magnitude + 1e-12)  # Add epsilon to avoid log(0)

# Frequency axis for STFT
freqs_stft = np.linspace(0, fs // 2, n_fft // 2 + 1)

# Find indices for 1900 Hz to 2100 Hz
freq_mask = (freqs_stft >= 1900) & (freqs_stft <= 2100)
freqs_zoom = freqs_stft[freq_mask]
log_magnitude_zoom = log_magnitude[freq_mask, :]

# Plot spectrogram (log-magnitude, 1900–2100 Hz)
plt.figure(figsize=(10, 4))
plt.imshow(log_magnitude_zoom, origin='lower', aspect='auto', cmap='viridis',
           extent=[0, Time[-1].item(), freqs_zoom[0], freqs_zoom[-1]])
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
frame_idx = log_magnitude_zoom.shape[1] // 2
axs[1].plot(freqs_zoom, log_magnitude_zoom[:, frame_idx])
axs[1].set_xlabel('Frequency [Hz]')
axs[1].set_ylabel('Log Magnitude [dB]')
axs[1].set_title(f'STFT Frame at time {Time[frame_idx*hop_length]:.4f} s')
axs[1].grid(True)

plt.tight_layout()
plt.show()
