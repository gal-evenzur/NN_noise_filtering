import numpy as np
import torch
from numpy.random import normal as normal

def var(reducer):
    # White noise will fall 95% of the time between +- 1.96 * sigma
    # Returns Variance for which White noise will fall 95% between +- reducer
    return reducer/1.96


def make_fft(data, t):
    L = len(t)
    X_w = abs(np.fft.fft(data)/L)
    X_w2 = X_w[0:int(L/2) + 3]
    X_w2 = 2 * X_w2[1:-1]

    d = t[1] - t[0]
    rate = 1/d
    freq = rate/L * np.arange(0, int(L/2) + 1)

    return freq, X_w2


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
    I0_r = normal(I0, var(I0 / 5))
    B0_r = B0 * normal(1, var(0.5)) * 10 ** (normal(0, var(2)))
    B0_r = abs(B0_r)
    F_B_r = normal(F_B, var(5))
    noise_strength_r = noise_strength * (1 + normal(1, var(1)))
    noise_strength_r = abs(noise_strength_r)
    pink_percentage = 0.4 + normal(0.3, var(0.3))

    return I0_r, B0_r, F_B_r, noise_strength_r, pink_percentage


def rand_test(I0, B0, F_B, noise_strength):
    I0_r = normal(I0, var(I0 / 4))
    B0_r = B0 * normal(1, var(1)) * 10 ** (normal(0, var(3)))
    B0_r = abs(B0_r)
    F_B_r = normal(F_B, var(5))
    noise_strength_r = noise_strength * (1 + normal(1, var(1)))
    noise_strength_r = abs(noise_strength_r)
    pink_percentage = 0.4 + normal(0.3, var(0.3))

    return I0_r, B0_r, F_B_r, noise_strength_r, pink_percentage

def Signal_Noise_FFts(I0, B0, F_B, noise_strength, pink_percentage):
    #I0 The current amplitude in the sensor[A]
    #B0 = The magnetic field on the sensor [T]
    #F_B  The magnetic field frequency [Hz]
    #noise_strength = 2.5e-5  # Noise will be 95% of times in this +-range

    #    CONSTS      #
    rho_perp = 2.7e-7  # [ohm * m]
    AMRR = 0.02  # AMR ratio (rho_par-rho_perp)/rho_perp
    B_k = 10e-4  # [T] Difference of resistence for parallel B vs Perp B
    # Isotope Field
    thickness = 200e-9  # [m] of the actual magnetic layer
    LL = 10e-6  # The misalignment[m]
    WW = 600e-6  # The width of the sensor [m]
    F_c = 2000  # The frequency of the current in the sensor [Hz]
    pi = np.pi

    #   GENERATING VOLTAGE TIME DOMAIN    #
    dt = 1e-4
    Time = np.arange(0, 1, dt)  # Starts at t=0 and ends at t=0.999s so it is repeating itself
    n = np.size(Time)
    delta_rho = rho_perp * AMRR  # [ohm * m]


    BB = B0 * np.sin(2 * pi * F_B * Time)
    II = I0 * np.sin(2 * pi * F_c * Time)
    hh = BB / B_k

    Voltage = (II / thickness) * ((rho_perp + delta_rho - delta_rho * (hh ** 2)) * (LL / WW) + delta_rho * hh)
    freq, fVolt = make_fft(Voltage, Time)

    #   CREATING NOISES     #
    Noise, *_ = make_noise(dt, noise_strength, pink_percentage)


    #   COMBINED SIGNAL   #
    Signal = Noise + Voltage
    _, fSignal = make_fft(Signal, Time)

    return Voltage, fVolt, Signal, fSignal, Time, freq

