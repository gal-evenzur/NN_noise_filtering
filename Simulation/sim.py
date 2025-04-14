import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from fft_pink_noise import *
import colorednoise
import json


#   CREATING TRAIN NOISE + SIGNAL  #
sig_time = []
sig_f = []

n_trains = 2
for _ in range(n_trains):
    I0 = 50e-3  # The current amplitude in the sensor[A]
    B0 = 1e-4  # The magnetic field on the sensor [T]
    F_B = 10  # The magnetic field frequency [Hz]
    noise_strength = 2.5e-5
    pink_percentage = 0.5
    Signal, fSignal, Time, freq = Signal_Noise_FFts(I0, B0, F_B, noise_strength, pink_percentage)

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
    B0 = 1e-4  # The magnetic field on the sensor [T]
    F_B = 10  # The magnetic field frequency [Hz]
    noise_strength = 2.5e-5
    pink_percentage = 0.5
    Signal, fSignal, __, ___ = Signal_Noise_FFts(I0, B0, F_B, noise_strength, pink_percentage)

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