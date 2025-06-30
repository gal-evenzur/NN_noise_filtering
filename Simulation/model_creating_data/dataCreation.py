import numpy as np

from fft_pink_noise import *
import json

add_signals = False

#   CREATING TRAIN NOISE + SIGNAL  #
clear_sig = []
clear_sig_f = []
sig_time = []
sig_f = []
Corresponding_B_strength = []
Corresponding_Main_Peak_strength = []
Corresponding_F_B = []


n_trains = 1000
for _ in range(n_trains):
    I0 = 50e-3  # The current amplitude in the sensor[A]
    B0 = 4e-12  # The magnetic field on the sensor [T]
    F_B = 15  # The magnetic field frequency [Hz]
    noise_strength = 0.5e-10
    # Create a tuple with all the randomized parameters
    params = rand_train(I0, B0, F_B,noise_strength)
    Volt, fVolt, Signal, fSignal, Time, freq = Signal_Noise_FFts(*params)

    real_F_B = params[2]

    if add_signals:
        clear_sig.append(Volt.tolist())
        sig_time.append(Signal.tolist())
    clear_sig_f.append(fVolt.tolist())
    sig_f.append(fSignal.tolist())
    Corresponding_F_B.append(real_F_B)

    # Corresponding_B_strength.append(fVolt[2000-real_F_B])
    # Corresponding_Main_Peak_strength.append(np.max(fVolt))

train = {
    "cSignal": clear_sig,
    "f_cSignal": clear_sig_f,
    "signals": sig_time,
    "f_signals": sig_f,
    "F_B": Corresponding_F_B,
}


n_validate = 200
for _ in range(n_validate):
    I0 = 50e-3  # The current amplitude in the sensor[A]
    B0 = 4e-12  # The magnetic field on the sensor [T]
    F_B = 15  # The magnetic field frequency [Hz]
    noise_strength = 0.5e-10
    # Create a tuple with all the randomized parameters
    params = rand_train(I0, B0, F_B,noise_strength)
    Volt, fVolt, Signal, fSignal, Time, freq = Signal_Noise_FFts(*params)

    real_F_B = params[2]

    if add_signals:
        clear_sig.append(Volt.tolist())
        sig_time.append(Signal.tolist())
    clear_sig_f.append(fVolt.tolist())
    sig_f.append(fSignal.tolist())
    Corresponding_F_B.append(real_F_B)

    # Corresponding_B_strength.append(fVolt[2000-real_F_B])
    # Corresponding_Main_Peak_strength.append(np.max(fVolt))

validate = {
    "cSignal": clear_sig,
    "f_cSignal": clear_sig_f,
    "signals": sig_time,
    "f_signals": sig_f,
    "F_B": Corresponding_F_B,
}


#   CREATING TEST NOISE AND SIGNAL    #
clear_sig = []
clear_sig_f = []
sig_time = []
sig_f = []
Corresponding_B_strength = []
Corresponding_Main_Peak_strength = []
Corresponding_F_B = []

n_tests = 200
for _ in range(n_tests):

    # The base parameters:
    I0 = 50e-3  # The current amplitude in the sensor[A]
    B0 = 4e-12  # The magnetic field on the sensor [T]
    F_B = 15  # The magnetic field frequency [Hz]
    noise_strength = 0.5e-10
    # Create a tuple with all the randomized parameters
    params = rand_test(I0, B0, F_B,noise_strength)
    Volt, fVolt, Signal, fSignal, Time, freq = Signal_Noise_FFts(*params)

    real_F_B = params[2]

    if add_signals:
        clear_sig.append(Volt.tolist())
        sig_time.append(Signal.tolist())
    clear_sig_f.append(fVolt.tolist())
    sig_f.append(fSignal.tolist())
    Corresponding_F_B.append(real_F_B)

    # Corresponding_B_strength.append(fVolt[2000-real_F_B])
    # Corresponding_Main_Peak_strength.append(np.max(fVolt))

test = {
    "cSignal": clear_sig,
    "f_cSignal": clear_sig_f,
    "signals": sig_time,
    "f_signals": sig_f,
    "F_B": Corresponding_F_B,
}

#   CREATING ONLY NOISE WITH SPECS OF TESTING #
clear_sig = []
clear_sig_f = []
sig_time = []
sig_f = []

n_noise = 10
for _ in range(10):

    params = rand_test(I0, B0, F_B,noise_strength)
    Volt, fVolt, Signal, fSignal, Time, freq = Signal_Noise_FFts(*params, only_noise=True)

    real_F_B = params[2]

    clear_sig_f.append(fVolt.tolist())
    sig_f.append(fSignal.tolist())


noise_for_test = {
    "f_cSignal": clear_sig_f,
    "f_noise": sig_f,
}

data = {
    "train": train,
    "validate": validate,
    "test": test,
    "noise_for_test": noise_for_test,
    "t": Time.tolist(),
    "f": freq.tolist()
}


#Writting to the data:
with open("data.json", "w") as f:
    json.dump(data, f, indent=4)