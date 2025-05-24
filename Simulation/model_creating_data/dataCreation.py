import math

from numpy.random import normal as normal
from fft_pink_noise import var as n_var
from fft_pink_noise import *
import json

add_signals = False

#   CREATING TRAIN NOISE + SIGNAL  #
clear_sig = []
clear_sig_f = []
sig_time = []
sig_f = []


n_trains = 1000
for _ in range(n_trains):
    I0 = 50e-3  # The current amplitude in the sensor[A]
    B0 = 5e-6  # The magnetic field on the sensor [T]
    F_B = 15  # The magnetic field frequency [Hz]
    noise_strength = 1e-5
    # Create a tuple with all the randomized parameters
    params = rand_train(I0, B0, F_B,noise_strength)
    Volt, fVolt, Signal, fSignal, Time, freq = Signal_Noise_FFts(*params)

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
    B0 = 5e-6  # The magnetic field on the sensor [T]
    F_B = 15  # The magnetic field frequency [Hz]
    noise_strength = 2.5e-5

    params = rand_test(I0, B0, F_B,noise_strength)
    Volt, fVolt, Signal, fSignal, Time, freq = Signal_Noise_FFts(*params)


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