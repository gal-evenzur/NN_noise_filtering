import math

from numpy.random import normal as normal
from fft_pink_noise import noise_variance as n_var
from fft_pink_noise import *
import json

add_signals = False

#   CREATING TRAIN NOISE + SIGNAL  #
clear_sig = []
clear_sig_f = []
sig_time = []
sig_f = []


n_trains = 800
for _ in range(n_trains):
    I0 = 50e-3  # The current amplitude in the sensor[A]
    I0_r = normal(I0, n_var(I0 / 5))
    B0 = 5e-6  # The magnetic field on the sensor [T]
    B0_r = B0 * normal(1, n_var(0.5)) * 10**(normal(0, n_var(2) ))
    F_B = 15  # The magnetic field frequency [Hz]
    F_B_r = normal(F_B, n_var(5))
    noise_strength = 1e-5
    noise_strength_r = noise_strength * (1 + normal(1, n_var(1)))
    pink_percentage = 0.4 + normal(0.3, n_var(0.3))
    Volt, fVolt, Signal, fSignal, Time, freq = Signal_Noise_FFts(I0_r, abs(B0_r), F_B_r, abs(noise_strength_r), pink_percentage)

    if add_signals:
        clear_sig.append(Volt.tolist())
        sig_time.append(Signal.tolist())
    clear_sig_f.append(fVolt.tolist())
    sig_f.append(fSignal.tolist())

train = {
    "cSignal": clear_sig,
    "f_cSignal": clear_sig_f,
    "signals": sig_time,
    "f_signals": sig_f,
}

#   CREATING TEST NOISE AND SIGNAL    #
clear_sig = []
clear_sig_f = []
sig_time = []
sig_f = []

n_tests = 200
for _ in range(n_tests):
    I0 = 50e-3  # The current amplitude in the sensor[A]
    I0_r = normal(I0, n_var(I0 / 4))
    B0 = 5e-6  # The magnetic field on the sensor [T]
    B0_r = B0 * normal(1, n_var(1)) * 10**(normal(0, n_var(3) ))
    F_B = 15  # The magnetic field frequency [Hz]
    F_B_r = normal(F_B, n_var(5))
    noise_strength = 2.5e-5
    noise_strength_r = noise_strength * (1 + normal(1, n_var(1)))
    pink_percentage = 0.4 + normal(0.3, n_var(0.3))
    Volt, fVolt, Signal, fSignal, Time, freq = Signal_Noise_FFts(I0_r, abs(B0_r), F_B_r, abs(noise_strength_r), pink_percentage)

    if add_signals:
        clear_sig.append(Volt.tolist())
        sig_time.append(Signal.tolist())
    clear_sig_f.append(fVolt.tolist())
    sig_f.append(fSignal.tolist())

test = {
    "cSignal": clear_sig,
    "f_cSignal": clear_sig_f,
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