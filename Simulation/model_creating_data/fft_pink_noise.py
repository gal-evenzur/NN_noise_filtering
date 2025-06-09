import random

import numpy as np
import torch
from numpy.random import normal as normal

def var(reducer):
    # White noise will fall 95% of the time between +- 1.96 * sigma
    # Returns Variance for which White noise will fall 95% between +- reducer
    return reducer/1.96

def make_fft(data, rate, num):
    Fs = rate  # Sampling frequency
    L = num  # Length of signal

    # Compute the Fourier transform of the signal
    Y = np.fft.fft(data)

    # Compute the two-sided spectrum P2. Then compute the single-sided spectrum P1
    # based on P2 and the even-valued signal length L
    P2 = np.abs(Y / L)
    P1 = P2[:L // 2 + 1]
    P1[1:-1] = 2 * P1[1:-1]  # Double the amplitudes except DC and Nyquist

    # Define the frequency domain f
    f = Fs * np.arange(0, L // 2 + 1) / L

    return f, P1

def make_pink_noise(t, sigma):
    L = len(t)
    d = t[1] - t[0]
    rate = 1/d
    white_noise = np.random.normal(0, sigma, L)

    fwNoise = np.fft.rfft(white_noise)
    freq = rate/L * np.arange(0, int(L/2) + 1)


    freq[0] = freq[1]  # Avoid division by zero at DC
    fwNoise = fwNoise / (freq)**0.5

    pink_noise = np.fft.irfft(fwNoise)

    # 6. Normalize to zero mean and unit variance (optional but useful)
    pink_noise -= np.mean(pink_noise)

    return pink_noise

def make_noise(dt, n_power, p_perc):
    Time = np.arange(0, 1, dt)  # Starts at t=0 and ends at t=0.999s so it is repeating itself
    n = np.size(Time)

    #   CREATING NOISES     #
    Noise_variance = var(n_power)

    wNoise = np.random.normal(0, Noise_variance, n)
    # _, fwNoise = make_fft(wNoise, Time)
    # _, S_wNoise = welch(wNoise, 1/dt, nperseg=n)

    # Using filtering in DFT Domain
    pNoise = make_pink_noise(Time, Noise_variance)
    # _, S_pNoise = welch(pNoise, 1/dt, nperseg=n)
    # _, fpNoise = make_fft(pNoise, Time)

    # Pink Using a library
    # pNoise = colorednoise.powerlaw_psd_gaussian(1, n) * Noise_var

    Noise = p_perc * pNoise + (1 - p_perc) * wNoise
    # _, fNoise = make_fft(Noise, Time)

    return Noise, pNoise, wNoise

def rand_train(I0, B0, F_B, noise_strength):
    I0_r = I0 * random.uniform(0.1, 5)
    B0_r = B0 * random.uniform(0.1, 10)
    F_B_r = random.randint(F_B - 5, F_B + 20)
    noise_strength_r = noise_strength*random.uniform(0.8, 1.4)
    pink_percentage = np.abs(normal(0, 0.5))

    return I0_r, B0_r, F_B_r, noise_strength_r, pink_percentage


def rand_test(I0, B0, F_B, noise_strength):
    I0_r = I0 + normal(0, var(0.1*I0))
    B0_r = B0 * random.uniform(0.25, 3)
    F_B_r = F_B
    noise_strength_r = noise_strength
    pink_percentage = 0

    return I0_r, B0_r, F_B_r, noise_strength_r, pink_percentage

def Signal_Noise_FFts(I0, B0, F_B, noise_strength, pink_percentage, is_padding=False, is_window=False):
    #I0 The current amplitude in the sensor[A]
    #B0 = The magnetic field on the sensor [T]
    #F_B  The magnetic field frequency [Hz]
    #noise_strength = 2.5e-5  # Noise will be 95% of times in this +-range

    #    CONSTS      #
    # Create a voltage signal from a PHE sensor
    rho_perp = 2.7e-7  # [ohm * m]
    AMRR = 0.02  # AMR ratio (rho_par-rho_perp)/rho_perp
    B_k = 10e-4  # [T] Difference of resistance for parallel B vs Perp B

    # Isotropy Field
    thickness = 200e-9  # [m] Thickness of magnetic layer
    LL = 10e-6  # The misalignment [m]
    WW = 600e-6  # The width of the sensor [m]
    F_c = 2000  # The frequency of the current in the sensor [Hz]
    dt = 1e-4  # [sec]

    # Time vector
    Time = np.arange(0, 1, dt)

    # Calculate voltage signal
    delta_rho = rho_perp * AMRR  # [ohm * m]
    BB = B0 * np.sin(2 * np.pi * F_B * Time)
    II = I0 * np.sin(2 * np.pi * F_c * Time)
    hh = BB / B_k

    Voltage = (II / thickness) * ((rho_perp + delta_rho - delta_rho * (hh ** 2)) * (LL / WW) + delta_rho * hh)

    if is_window:
        # Window the signal to prevent spectal leakage when padding
        window = np.hanning(len(Voltage))
        Voltage = window * Voltage

    if is_padding:
        # Padding so the peaks aren't as sharp
        n = np.ceil(np.log2(len(Voltage)))
        Voltage = np.pad(Voltage, (0,len(Voltage)*4))


    f, P1 = make_fft(Voltage, 1 / dt, len(Voltage))

    #   CREATING NOISES     #
    Noise, *_ = make_noise(dt, noise_strength, pink_percentage)


    #   COMBINED SIGNAL   #
    Signal = Noise + Voltage

    if is_window:
        # Window the signal to prevent spectal leakage when padding
        window = np.hanning(len(Signal))
        Signal = window * Signal

    if is_padding:
        # Padding so the peaks aren't as sharp
        n = np.ceil(np.log2(len(Signal)))
        Signal = np.pad(Signal, (0,len(Signal)*4))

    _, fSignal = make_fft(Signal, 1/dt, len(Signal))

    return Voltage, P1, Signal, fSignal, Time, f

def peak_heights(clear_signal, f_b, f_center, dir=False):
    # Receives a Tensor, and returns the height of each of the peaks in the signal
    # Needs to receive the clean signals in a dataset form (n_samps x L)
    if type(clear_signal) == np.ndarray:
        freqs = [f_center - f_b, f_center, f_center + f_b]
        return clear_signal[freqs]

    n_samp = len(clear_signal)

    l = f_center - f_b
    m = torch.full_like(l, 2000)
    r = f_center + f_b

    # Concatenate indices: shape (n_samp, 3)
    indices = torch.cat([l, m, r], dim=1)  # (n_samp, 3)

    # Create row indices: [0, 1, ..., n_samp-1] → shape (n_samp, 1), then expand to (n_samp, 3)
    row_indices = torch.arange(n_samp).unsqueeze(1).expand(-1, 3)

    # Use advanced indexing to extract the values
    result = clear_signal[row_indices, indices]  # shape: (n_samp, 3)

    peaks = {
        'left': result[:, 0],
        'center': result[:, 1],
        'right': result[:, 2],
    }

    if dir:
        return peaks
    else:
        return result


