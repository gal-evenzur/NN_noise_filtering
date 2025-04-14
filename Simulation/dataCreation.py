import math

from numpy.random import normal as normal
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from fft_pink_noise import noise_variance as n_var
from fft_pink_noise import *
import colorednoise
import json


#   CREATING TRAIN NOISE + SIGNAL  #
sig_time = []
sig_f = []


n_trains = 2
for _ in range(n_trains):
    I0 = 50e-3  # The current amplitude in the sensor[A]
    I0_r = normal(I0, n_var(I0 / 5))
    B0 = 5e-6  # The magnetic field on the sensor [T]
    B0_r = B0 * normal(1, n_var(0.5)) * 10**(normal(0, n_var(2) ))
    F_B = 15  # The magnetic field frequency [Hz]
    F_B_r = normal(F_B, n_var(5))
    noise_strength = 2.5e-5
    noise_strength_r = noise_strength * normal(1, n_var(1))
    pink_percentage = 0.4 + normal(0.3, n_var(0.3))
    Signal, fSignal, Time, freq = Signal_Noise_FFts(I0_r, abs(B0_r), F_B_r, noise_strength_r, pink_percentage)

    sig_time.append(Signal.tolist())
    sig_f.append(fSignal.tolist())

train = {
    "signals": sig_time,
    "f_signals": sig_f,
}

#   CREATING TEST NOISE AND SIGNAL    #
sig_time = []
sig_f = []
n_tests = 2
for _ in range(n_tests):
    I0 = 50e-3  # The current amplitude in the sensor[A]
    I0_r = normal(I0, n_var(I0 / 4))
    B0 = 5e-6  # The magnetic field on the sensor [T]
    B0_r = B0 * normal(1, n_var(1)) * 10**(normal(0, n_var(3) ))
    F_B = 15  # The magnetic field frequency [Hz]
    F_B_r = normal(F_B, n_var(5))
    noise_strength = 2.5e-5
    noise_strength_r = noise_strength * normal(1, n_var(1))
    pink_percentage = 0.4 + normal(0.3, n_var(0.3))
    Signal, fSignal, Time, freq = Signal_Noise_FFts(I0_r, abs(B0_r), F_B_r, noise_strength_r, pink_percentage)

    sig_time.append(Signal.tolist())
    sig_f.append(fSignal.tolist())

test = {
    "signals": sig_time,
    "f_signals": sig_f,
}

data = {
    "train": train,
    "test": test,
    "t": Time.tolist(),
    "f": freq.tolist()
}

#Writting to the data:
with open("data.json", "w") as f:
    json.dump(data, f, indent=4)