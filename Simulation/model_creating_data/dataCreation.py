from fft_pink_noise import *
import json

add_signals = False

#   CREATING TRAIN NOISE + SIGNAL  #
clear_sig = []
clear_sig_f = []
sig_time = []
sig_f = []
Corresponding_B_strength = []
Corresponding_F_B = []


n_trains = 1000
for _ in range(n_trains):
    I0 = 50e-3  # The current amplitude in the sensor[A]
    B0 = 1e-9  # The magnetic field on the sensor [T]
    F_B = 15  # The magnetic field frequency [Hz]
    noise_strength = 1e-7
    # Create a tuple with all the randomized parameters
    params = rand_train(I0, B0, F_B,noise_strength)
    Volt, fVolt, Signal, fSignal, Time, freq = Signal_Noise_FFts(*params)

    if add_signals:
        clear_sig.append(Volt.tolist())
        sig_time.append(Signal.tolist())
    clear_sig_f.append(fVolt.tolist())
    Corresponding_B_strength.append(params[1])
    Corresponding_F_B.append(params[2])
    sig_f.append(fSignal.tolist())

train = {
    "cSignal": clear_sig,
    "f_cSignal": clear_sig_f,
    "signals": sig_time,
    "f_signals": sig_f,
    "amplitudes": Corresponding_B_strength,
    "F_B": Corresponding_F_B,
}

#   CREATING TEST NOISE AND SIGNAL    #
clear_sig = []
clear_sig_f = []
sig_time = []
sig_f = []
Corresponding_B_strength = []

n_tests = 200
for _ in range(n_tests):
    I0 = 50e-3  # The current amplitude in the sensor[A]
    B0 = 1e-9  # The magnetic field on the sensor [T]
    F_B = 15  # The magnetic field frequency [Hz]
    noise_strength = 1e-7
    # Create a tuple with all the randomized parameters
    params = rand_train(I0, B0, F_B,noise_strength)
    Volt, fVolt, Signal, fSignal, Time, freq = Signal_Noise_FFts(*params)

    if add_signals:
        clear_sig.append(Volt.tolist())
        sig_time.append(Signal.tolist())
    clear_sig_f.append(fVolt.tolist())
    Corresponding_B_strength.append(params[1])
    Corresponding_F_B.append(params[2])
    sig_f.append(fSignal.tolist())

test = {
    "cSignal": clear_sig,
    "f_cSignal": clear_sig_f,
    "signals": sig_time,
    "f_signals": sig_f,
    "amplitudes": Corresponding_B_strength,
    "F_B": Corresponding_F_B,
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